"""V2回测: 测试集得分 + 5月回测"""
import sys, os, json
sys.path.insert(0, 'code/src')
import pandas as pd, numpy as np
import torch, joblib
from tqdm import tqdm
import multiprocessing as mp
from ensemble_config import *
from ensemble_models import *
from utils import engineer_features_39, engineer_features_158plus39
from predict import feature_cloums_map, feature_engineer_func_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load config
with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json')) as f:
    cfg = json.load(f)
sid2idx = cfg['stockid2idx']
expert_cfgs = cfg['expert_configs']
seq_len = cfg['sequence_length']
feat_num = cfg['feature_num']
mc_samples = cfg['mc_samples']
fc = feature_cloums_map[feat_num]
fe = feature_engineer_func_map[feat_num]
num_stocks = cfg['num_stocks']

def create_model(ecfg, input_dim):
    t = ecfg.get('type','transformer')
    if t == 'transformer': return StockTransformerExpert(input_dim, ecfg, num_stocks)
    if t == 'month_seasonal': return MonthSeasonalExpert(input_dim, ecfg, num_stocks)
    if t == 'aggressive': return AggressiveExpert(input_dim, ecfg, num_stocks)
    if t == 'brownian': return BrownianNoiseExpert(input_dim, ecfg, num_stocks)
    if t == 'statarb': return StatArbRegressionExpert(input_dim, ecfg, num_stocks)
    if t == 'conv': return ConvStockExpert(input_dim, ecfg, num_stocks)
    return None

# Load all data
raw = pd.read_csv('data/train.csv', dtype={'股票代码': str})
raw['日期'] = pd.to_datetime(raw['日期'], format='mixed')

# ====== TEST SET (March 2026) ======
print('\n' + '='*60)
print('回测: 官方测试集 (2026-03-09 ~ 03-13)')
print('='*60)

train_cutoff = pd.Timestamp('2026-03-06')
train_df = raw[raw['日期'] <= train_cutoff].copy()
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})

print(f'训练截止: {train_df["日期"].max()}')
print(f'测试集: {test_df["日期"].min()} ~ {test_df["日期"].max()}')

# Feature engineering
df = train_df.sort_values(['股票代码','日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _,g in df.groupby('股票代码',sort=False) if len(g) >= seq_len+10]
with mp.Pool(min(8, mp.cpu_count())) as pool:
    plist = list(tqdm(pool.imap(fe, groups), total=len(groups), desc='FE'))
p = pd.concat(plist).reset_index(drop=True)
p['instrument'] = p['股票代码'].map(sid2idx)
p = p.dropna(subset=['instrument'])
p['instrument'] = p['instrument'].astype(np.int64)
p['日期'] = pd.to_datetime(p['日期'], format='mixed')
p[fc] = p[fc].replace([np.inf,-np.inf], np.nan).fillna(0.0)

# Use the saved scaler
scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
p[fc] = scaler.transform(p[fc])

# Build sequences
latest = train_df['日期'].max()
stock_ids = sorted(train_df['股票代码'].unique())
seqs, seq_ids = [], []
for sid in stock_ids:
    hist = p[(p['股票代码']==sid)&(p['日期']<=latest)].sort_values('日期').tail(seq_len)
    if len(hist) == seq_len:
        seqs.append(hist[fc].values.astype(np.float32))
        seq_ids.append(sid)

x = torch.from_numpy(np.asarray(seqs, dtype=np.float32)).unsqueeze(0).to(device)
input_dim = fc.shape[0] if hasattr(fc, 'shape') else len(fc)
input_dim = seqs[0].shape[-1]  # actual dim
print(f'有效股票: {len(seq_ids)}, dim: {input_dim}')

# Load experts
print('加载专家...')
experts = []
for ecfg in expert_cfgs:
    path = os.path.join(OUTPUT_DIR, f'expert_{ecfg["name"]}.pth')
    if not os.path.exists(path): continue
    m = create_model(ecfg, input_dim)
    if m is None: continue
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    experts.append(m)
print(f'{len(experts)} experts')

# MC prediction
print(f'MC x{mc_samples}...')
all_scores = []
for e in experts:
    e.train()
    mc = []
    with torch.no_grad():
        for _ in range(mc_samples):
            mc.append(e(x).squeeze(0))
    all_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())

# Meta
meta_path = os.path.join(OUTPUT_DIR, 'meta_aggregator.pth')
if os.path.exists(meta_path):
    meta = MetaAggregator(len(experts), num_stocks).to(device)
    meta.load_state_dict(torch.load(meta_path, map_location=device))
    meta.eval()
    es = torch.from_numpy(np.stack(all_scores, axis=-1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        final = meta(es).squeeze(0).cpu().numpy()
else:
    final = np.mean(all_scores, axis=0)

# Smart weights
order = np.argsort(final)[::-1]
top_n = 5
top_scores = final[order[:top_n]]
top_ids = [seq_ids[i] for i in order[:top_n]]

temperature = 0.3
score_weights = np.exp(top_scores / temperature)
score_weights /= score_weights.sum()

# 重仓上限30%
MAX_SINGLE = 0.30
for _ in range(10):
    overflow = 0; capped = 0
    for i in range(top_n):
        if score_weights[i] > MAX_SINGLE:
            overflow += score_weights[i] - MAX_SINGLE
            score_weights[i] = MAX_SINGLE
            capped += 1
    if overflow > 0 and capped < top_n:
        uc = top_n - capped
        for i in range(top_n):
            if score_weights[i] < MAX_SINGLE:
                score_weights[i] += overflow / uc
    else:
        break

max_score = top_scores[0]
mean_score = final.mean()
std_score = final.std()
confidence = (max_score - mean_score) / (std_score + 1e-8)

if confidence < 1.0:
    pos_ratio = 0.3
elif confidence < 2.0:
    pos_ratio = 0.5 + (confidence - 1.0) * 0.25
else:
    pos_ratio = min(1.0, 0.75 + (confidence - 2.0) * 0.1)

final_weights = score_weights * pos_ratio

print(f'置信度: {confidence:.2f}σ, 仓位: {pos_ratio:.0%}')
print(f'\n预测Top5:')
for i in range(top_n):
    print(f'  {top_ids[i]}: 得分{top_scores[i]:.4f}, 权重{final_weights[i]:.4f}')
print(f'  空仓: {1-final_weights.sum():.4f}')

# Evaluate on test set
print(f'\n=== 测试集实际收益 ===')
test_df['股票代码'] = test_df['股票代码'].astype(str)
pred_set = set(top_ids)
test_set = set(test_df['股票代码'].unique())
common = pred_set & test_set

if not common:
    print(f'FAIL: 预测股票不在测试集!')
    print(f'预测: {top_ids}')
    print(f'测试示例: {list(test_set)[:10]}')
else:
    tf = test_df[test_df['股票代码'].isin(top_ids)].groupby('股票代码').tail(5)
    def cr(g):
        return (g.iloc[-1]['开盘'] - g.iloc[0]['开盘']) / g.iloc[0]['开盘']
    rets = tf.groupby('股票代码').apply(cr).reset_index().rename(columns={0:'收益率'})
    res_df = pd.DataFrame({'股票代码': top_ids})
    res_df = res_df.merge(rets, on='股票代码', how='left')
    res_df['权重'] = final_weights[:len(res_df)]
    res_df['收益率'] = res_df['收益率'].fillna(0)
    fs = (res_df['收益率'] * res_df['权重']).sum()

    print(f'{"股票":>8s}  {"收益":>8s}  {"权重":>8s}  {"贡献":>8s}')
    for _, r in res_df.iterrows():
        contrib = r['收益率'] * r['权重']
        print(f'{r["股票代码"]:>8s}  {r["收益率"]:>+7.4%}  {r["权重"]:>7.4f}  {contrib:>+8.4%}')
    print(f'\n===== 综合得分: {fs:.6f} = {fs:.4%} =====')

# Also do May backtest
print(f'\n{"="*60}')
print('回测: 2026年5月')
print('='*60)
may_cutoff = pd.Timestamp('2026-04-30')
may_train = raw[raw['日期'] <= may_cutoff].copy()
may_test = raw[(raw['日期'] > may_cutoff) & (raw['日期'] <= pd.Timestamp('2026-05-10'))].copy()
may_dates = sorted(may_test['日期'].unique())[:5]

# Quick FE for May
df2 = may_train.sort_values(['股票代码','日期']).reset_index(drop=True)
groups2 = [g.reset_index(drop=True) for _,g in df2.groupby('股票代码',sort=False) if len(g) >= seq_len+10]
with mp.Pool(min(8, mp.cpu_count())) as pool:
    plist2 = list(tqdm(pool.imap(fe, groups2), total=len(groups2), desc='FE-May'))
p2 = pd.concat(plist2).reset_index(drop=True)
p2['instrument'] = p2['股票代码'].map(sid2idx)
p2 = p2.dropna(subset=['instrument'])
p2['instrument'] = p2['instrument'].astype(np.int64)
p2['日期'] = pd.to_datetime(p2['日期'], format='mixed')
p2[fc] = p2[fc].replace([np.inf,-np.inf], np.nan).fillna(0.0)
p2[fc] = scaler.transform(p2[fc])

latest2 = may_train['日期'].max()
stock_ids2 = sorted(may_train['股票代码'].unique())
seqs2, seq_ids2 = [], []
for sid in stock_ids2:
    hist = p2[(p2['股票代码']==sid)&(p2['日期']<=latest2)].sort_values('日期').tail(seq_len)
    if len(hist) == seq_len:
        seqs2.append(hist[fc].values.astype(np.float32))
        seq_ids2.append(sid)

x2 = torch.from_numpy(np.asarray(seqs2, dtype=np.float32)).unsqueeze(0).to(device)

all_scores2 = []
for e in experts:
    e.train()
    mc = []
    with torch.no_grad():
        for _ in range(mc_samples):
            mc.append(e(x2).squeeze(0))
    all_scores2.append(torch.stack(mc).mean(dim=0).cpu().numpy())

es2 = torch.from_numpy(np.stack(all_scores2, axis=-1)).unsqueeze(0).float().to(device)
with torch.no_grad():
    final2 = meta(es2).squeeze(0).cpu().numpy()

order2 = np.argsort(final2)[::-1]
top_ids2 = [seq_ids2[i] for i in order2[:5]]
top_scores2 = final2[order2[:5]]

score_weights2 = np.exp(top_scores2 / temperature)
score_weights2 /= score_weights2.sum()
pos2 = min(1.0, pos_ratio)
final_weights2 = score_weights2 * pos2

may_data = may_test[may_test['日期'].isin(may_dates[:5])]
print(f'5月交易日: {[d.date() for d in may_dates[:5]]}')

print(f'\n{"股票":>8s}  {"收益":>8s}  {"权重":>8s}  {"贡献":>8s}')
total = 0
for i in range(5):
    sid = top_ids2[i]
    w = final_weights2[i]
    stock_may = may_data[may_data['股票代码']==sid].sort_values('日期')
    if len(stock_may) >= 2:
        ret = (float(stock_may.iloc[-1]['开盘']) - float(stock_may.iloc[0]['开盘'])) / float(stock_may.iloc[0]['开盘'])
    else:
        ret = 0
    contrib = ret * w
    total += contrib
    print(f'{sid:>8s}  {ret:>+7.4%}  {w:>7.4f}  {contrib:>+8.4%}')

print(f'\n===== 5月综合: {total:.6f} = {total:.4%} =====')
