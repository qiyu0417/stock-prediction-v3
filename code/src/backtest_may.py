"""回测2026年5月——用4月底数据预测，对比5月前5个交易日实际收益"""
import sys, os, json
sys.path.insert(0, 'code/src')
import pandas as pd, numpy as np
import torch, joblib
from tqdm import tqdm
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from ensemble_config import *
from ensemble_models import *
from utils import engineer_features_39, engineer_features_158plus39
from predict import feature_cloums_map, feature_engineer_func_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 加载配置
with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json')) as f:
    cfg = json.load(f)
sid2idx = cfg['stockid2idx']
expert_cfgs = cfg['expert_configs']
seq_len = cfg['sequence_length']
feat_num = cfg['feature_num']
input_dim = cfg['input_dim']
mc_samples = cfg['mc_samples']
fc = feature_cloums_map[feat_num]
fe = feature_engineer_func_map[feat_num]
num_stocks = cfg['num_stocks']

# 加载全量数据
raw = pd.read_csv('data/train.csv', dtype={'股票代码': str})
raw['日期'] = pd.to_datetime(raw['日期'], format='mixed')

# 切分：4月30日之前为训练，5月前5个交易日为测试
train_cutoff = pd.Timestamp('2026-04-30')
train_df = raw[raw['日期'] <= train_cutoff].copy()
may_df = raw[(raw['日期'] > train_cutoff) & (raw['日期'] <= pd.Timestamp('2026-05-10'))].copy()

print(f"训练截止: {train_df['日期'].max()}")
print(f"5月数据: {may_df['日期'].min()} ~ {may_df['日期'].max()}")
print(f"5月交易天数: {may_df['日期'].nunique()}")

# 找到5月第一个有数据的交易日作为预测起始日
may_dates = sorted(may_df['日期'].unique())
may_first_5 = may_dates[:5]
print(f"5月前5个交易日: {[d.date() for d in may_first_5]}")

# 特征工程
print("\n特征工程...")
df = train_df.sort_values(['股票代码','日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _,g in df.groupby('股票代码',sort=False) if len(g) >= seq_len+10]
with mp.Pool(min(8, mp.cpu_count())) as pool:
    plist = list(tqdm(pool.imap(fe, groups), total=len(groups), desc='特征'))
p = pd.concat(plist).reset_index(drop=True)
p['instrument'] = p['股票代码'].map(sid2idx)
p = p.dropna(subset=['instrument'])
p['instrument'] = p['instrument'].astype(np.int64)
p['日期'] = pd.to_datetime(p['日期'], format='mixed')
p[fc] = p[fc].replace([np.inf,-np.inf], np.nan).fillna(0.0)

scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
p[fc] = scaler.transform(p[fc])

# 构建推理序列（用4月30日最新数据）
latest = train_df['日期'].max()
stock_ids = sorted(train_df['股票代码'].unique())
seqs, seq_ids = [], []
for sid in stock_ids:
    hist = p[(p['股票代码']==sid)&(p['日期']<=latest)].sort_values('日期').tail(seq_len)
    if len(hist) == seq_len:
        seqs.append(hist[fc].values.astype(np.float32))
        seq_ids.append(sid)

x = torch.from_numpy(np.asarray(seqs, dtype=np.float32)).unsqueeze(0).to(device)
print(f"有效股票: {len(seq_ids)}")

# 加载专家
def create_model(ecfg):
    t = ecfg.get('type','transformer')
    if t == 'transformer': return StockTransformerExpert(input_dim, ecfg, num_stocks)
    if t == 'month_seasonal': return MonthSeasonalExpert(input_dim, ecfg, num_stocks)
    if t == 'conv': return ConvStockExpert(input_dim, ecfg, num_stocks)
    return None

experts = []
for ecfg in expert_cfgs:
    path = os.path.join(OUTPUT_DIR, f'expert_{ecfg["name"]}.pth')
    if not os.path.exists(path): continue
    m = create_model(ecfg)
    if m is None: continue
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    experts.append(m)

# MC推理
print(f"MC推理 x{mc_samples}...")
all_scores = []
for e in experts:
    e.train()
    mc = []
    with torch.no_grad():
        for _ in range(mc_samples):
            mc.append(e(x).squeeze(0))
    all_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())

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

order = np.argsort(final)[::-1]
top5 = [seq_ids[i] for i in order[:5]]
print(f"\n预测Top5: {top5}")

# 计算5月实际收益
print("\n=== 5月实际表现 ===")
may_df_5 = may_df[may_df['日期'].isin(may_first_5)]
may_df_5 = may_df_5[may_df_5['股票代码'].isin(top5)]

if len(may_df_5) == 0:
    print("无匹配的5月数据!")
    # Try without filtering by dates - just use first 5 days per stock
    may_pred = may_df[may_df['股票代码'].isin(top5)].copy()
    for sid in top5:
        stock_may = may_df[may_df['股票代码']==sid].sort_values('日期')
        if len(stock_may) >= 5:
            s = stock_may.iloc[0]
            e = stock_may.iloc[4]  # 5th trading day
            ret = (float(e['开盘']) - float(s['开盘'])) / float(s['开盘'])
            print(f"  {sid}: {ret:+.4%}")
else:
    def calc_return(g):
        s = g.iloc[0]; e = g.iloc[-1]
        return (float(e['开盘']) - float(s['开盘'])) / float(s['开盘'])

    total_return = 0
    for sid in top5:
        stock_data = may_df_5[may_df_5['股票代码']==sid].sort_values('日期')
        if len(stock_data) >= 2:
            ret = calc_return(stock_data)
            total_return += ret * 0.2
            print(f"  {sid}: {ret:+.4%} ({len(stock_data)}天)")
        else:
            print(f"  {sid}: 数据不足")
    print(f"\n综合得分: {total_return:.6f} = {total_return:.4%}")
