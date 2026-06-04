"""智能权重分配：按置信度加权 + 空仓机制"""
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

# 数据
raw = pd.read_csv('data/train.csv', dtype={'股票代码': str})
raw['日期'] = pd.to_datetime(raw['日期'], format='mixed')
latest = raw['日期'].max()
print(f"最新日期: {latest.date()}")

# 特征工程
df = raw.sort_values(['股票代码','日期']).reset_index(drop=True)
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

stock_ids = sorted(raw['股票代码'].unique())
seqs, seq_ids = [], []
for sid in stock_ids:
    hist = p[(p['股票代码']==sid)&(p['日期']<=latest)].sort_values('日期').tail(seq_len)
    if len(hist) == seq_len:
        seqs.append(hist[fc].values.astype(np.float32))
        seq_ids.append(sid)

x = torch.from_numpy(np.asarray(seqs, dtype=np.float32)).unsqueeze(0).to(device)
print(f"有效: {len(seq_ids)}只")

# 加载专家
def create_model(ecfg):
    t = ecfg.get('type','transformer')
    if t == 'transformer': return StockTransformerExpert(input_dim, ecfg, num_stocks)
    if t == 'month_seasonal': return MonthSeasonalExpert(input_dim, ecfg, num_stocks)
    if t == 'aggressive': return AggressiveExpert(input_dim, ecfg, num_stocks)
    if t == 'brownian': return BrownianNoiseExpert(input_dim, ecfg, num_stocks)
    if t == 'statarb': return StatArbRegressionExpert(input_dim, ecfg, num_stocks)
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
print(f"MC x{mc_samples}...")
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

# ===== 智能权重分配 =====
order = np.argsort(final)[::-1]
top_n = 5

# 取Top5分数
top_scores = final[order[:top_n]]
top_ids = [seq_ids[i] for i in order[:top_n]]

# Softmax转权重（温度控制集中度）
temperature = 0.3
score_weights = np.exp(top_scores / temperature)
score_weights = score_weights / score_weights.sum()

# ===== 重仓上限：单票不超过30%，超过部分按比例分给其他 =====
MAX_SINGLE_WEIGHT = 0.30
for _ in range(10):  # 迭代直到所有权重<=上限
    overflow = 0
    capped = 0
    for i in range(top_n):
        if score_weights[i] > MAX_SINGLE_WEIGHT:
            overflow += score_weights[i] - MAX_SINGLE_WEIGHT
            score_weights[i] = MAX_SINGLE_WEIGHT
            capped += 1
    if overflow > 0 and capped < top_n:
        # 把超出部分分给未封顶的
        uncapped_count = top_n - capped
        for i in range(top_n):
            if score_weights[i] < MAX_SINGLE_WEIGHT:
                score_weights[i] += overflow / uncapped_count
    else:
        break

# 空仓逻辑
max_score = top_scores[0]
mean_score = final.mean()
std_score = final.std()
confidence = (max_score - mean_score) / (std_score + 1e-8)

if confidence < 1.0:
    position_ratio = 0.3
elif confidence < 2.0:
    position_ratio = 0.5 + (confidence - 1.0) * 0.25
else:
    position_ratio = min(1.0, 0.75 + (confidence - 2.0) * 0.1)

final_weights = score_weights * position_ratio

print(f"\n置信度: {confidence:.2f}σ, 仓位: {position_ratio:.0%}")
print(f"\n{'='*50}")
print(f"预测日期: {latest.date()}")
print(f"{'='*50}")
print(f"{'股票':>8s}  {'分数':>8s}  {'权重':>8s}")
print('-'*30)
total_w = 0
for i in range(top_n):
    w = final_weights[i]
    total_w += w
    print(f"{top_ids[i]:>8s}  {top_scores[i]:8.4f}  {w:7.4f}")
print(f"{'空仓':>8s}  {'':>8s}  {1-total_w:7.4f}")
print(f"\n总仓位: {total_w:.4f}, 留现金: {1-total_w:.4f}")

# 保存
os.makedirs('./output/', exist_ok=True)
result = pd.DataFrame({
    'stock_id': top_ids,
    'weight': [round(w, 4) for w in final_weights]
})
result.to_csv('./output/result.csv', index=False)
print("已保存 output/result.csv")
