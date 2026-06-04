"""Debug and fix prediction"""
import sys, os, json
sys.path.insert(0, 'code/src')
import pandas as pd
import torch
import numpy as np
from ensemble_config import *

# 1. Check raw data
raw = pd.read_csv('data/train.csv', dtype={'股票代码': str})
raw['日期'] = pd.to_datetime(raw['日期'], format='mixed')
codes = sorted(raw['股票代码'].unique())
latest = raw['日期'].max()
print(f'数据: {len(raw)}行, {len(codes)}股, 最新: {latest.date()}')
print(f'股票代码: {codes[:10]}...{codes[-3:]}')

# 2. Check config
with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json')) as f:
    cfg = json.load(f)
sid2idx = cfg['stockid2idx']
c2 = sorted(sid2idx.keys())
print(f'config codes: {c2[:10]}...{c2[-3:]}')
print(f'Match: {codes == c2}')

# 3. Build seq_ids (what prediction would use)
stock_ids = sorted(raw['股票代码'].unique())
seq_len = cfg['sequence_length']
seq_ids = []
for sid in stock_ids:
    hist = raw[(raw['股票代码']==sid)&(raw['日期']<=latest)].sort_values('日期').tail(seq_len)
    if len(hist) == seq_len:
        seq_ids.append(sid)
print(f'有效股票: {len(seq_ids)}, 示例: {seq_ids[:10]}')

# 4. Now the correct prediction
from ensemble_models import *
from utils import engineer_features_39, engineer_features_158plus39
from predict import feature_cloums_map, feature_engineer_func_map
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from tqdm import tqdm
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feat_num = cfg['feature_num']
fc = feature_cloums_map[feat_num]
mc_samples = cfg['mc_samples']
input_dim = cfg['input_dim']

# Feature engineering
print('\n特征工程...')
fe = feature_engineer_func_map[feat_num]
df = raw.sort_values(['股票代码','日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _,g in df.groupby('股票代码',sort=False)]
with mp.Pool(min(8, mp.cpu_count())) as pool:
    plist = list(tqdm(pool.imap(fe, groups), total=len(groups), desc='FE'))
p = pd.concat(plist).reset_index(drop=True)
p['instrument'] = p['股票代码'].map(sid2idx)
p = p.dropna(subset=['instrument'])
p['instrument'] = p['instrument'].astype(np.int64)
p['日期'] = pd.to_datetime(p['日期'])
p[fc] = p[fc].replace([np.inf,-np.inf], np.nan).fillna(0.0)

scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
p[fc] = scaler.transform(p[fc])

# Build sequences
seqs, seq_ids_final = [], []
for sid in stock_ids:
    hist = p[(p['股票代码']==sid)&(p['日期']<=latest)].sort_values('日期').tail(seq_len)
    if len(hist) == seq_len:
        seqs.append(hist[fc].values.astype(np.float32))
        seq_ids_final.append(sid)

x = torch.from_numpy(np.asarray(seqs, dtype=np.float32)).unsqueeze(0).to(device)
print(f'推理: {len(seq_ids_final)}只股票')

# Load experts
expert_configs = cfg['expert_configs']
num_stocks = cfg['num_stocks']

def create_model(ecfg):
    t = ecfg.get('type','transformer')
    if t == 'transformer': return StockTransformerExpert(input_dim, ecfg, num_stocks)
    if t == 'month_seasonal': return MonthSeasonalExpert(input_dim, ecfg, num_stocks)
    if t == 'conv': return ConvStockExpert(input_dim, ecfg, num_stocks)
    return None

print('加载专家...')
experts = []
for ecfg in expert_configs:
    path = os.path.join(OUTPUT_DIR, f'expert_{ecfg["name"]}.pth')
    if not os.path.exists(path): continue
    m = create_model(ecfg)
    if m is None: continue
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    experts.append(m)
    print(f'  {ecfg["name"]}')

# MC prediction
print(f'MC推理 x{mc_samples}...')
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
    meta = MetaAggregator(len(experts), num_stocks, hidden_dim=64).to(device)
    meta.load_state_dict(torch.load(meta_path, map_location=device))
    meta.eval()
    es = torch.from_numpy(np.stack(all_scores, axis=-1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        final = meta(es).squeeze(0).cpu().numpy()
else:
    final = np.mean(all_scores, axis=0)

# Top5
order = np.argsort(final)[::-1]
print(f'seq_ids_final size: {len(seq_ids_final)}, samples: {seq_ids_final[:10]}...{seq_ids_final[-3:]}')
print(f'final shape: {final.shape}, order[:5]: {order[:5]}')
for oi in order[:10]:
    print(f'  rank {oi}: seq_ids[{oi}] = {seq_ids_final[oi]}')
top5 = [seq_ids_final[i] for i in order[:5]]
print(f'\nTop5: {top5}')

os.makedirs('./output/', exist_ok=True)
result_df = pd.DataFrame({'stock_id': [str(s) for s in top5], 'weight': [0.2]*5})
result_df.to_csv('./output/result.csv', index=False)
print(f'结果已保存: {result_df["stock_id"].tolist()}')

# Evaluate on test
print('\n=== 测试集评估 ===')
td = pd.read_csv('data/test.csv', dtype={'股票代码': str})
od = pd.read_csv('output/result.csv')
od = od.rename(columns={'stock_id': '股票代码', 'weight': '权重'})

print(f'预测: {od["股票代码"].tolist()}')
print(f'在测试集中: {od["股票代码"].isin(td["股票代码"]).tolist()}')

if od['股票代码'].isin(td['股票代码']).all():
    tf = td[td['股票代码'].isin(od['股票代码'])].groupby('股票代码').tail(5)
    def cr(g):
        return (g.iloc[-1]['开盘'] - g.iloc[0]['开盘']) / g.iloc[0]['开盘']
    rets = tf.groupby('股票代码').apply(cr).reset_index().rename(columns={0:'收益率'})
    res = rets.merge(od, on='股票代码')
    fs = (res['收益率'] * res['权重']).sum()
    print()
    for _,r in res.iterrows():
        print(f'  {r["股票代码"]}: {r["收益率"]:+.4%}')
    print(f'\n===== 综合得分: {fs:.6f} = {fs:.4%} =====')
else:
    print('FAIL: 股票代码不匹配!')
