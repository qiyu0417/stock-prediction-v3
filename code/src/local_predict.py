"""本地V1模型预测 - 使用model/v1_ensemble/中的模型"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import torch, joblib, numpy as np, pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler

from ensemble_config import *
from ensemble_models import StockTransformerExpert, MonthSeasonalExpert, MetaAggregator
from utils import engineer_features_39, engineer_features_158plus39
from predict import feature_cloums_map, feature_engineer_func_map
from train import _build_label_and_clean

# 路径 - 相对于项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model', 'v1_ensemble')
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'train_updated.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'output', 'result2_v1.csv')

device = torch.device('cpu')
print(f'设备: {device}')

# 加载配置
with open(os.path.join(MODEL_DIR, 'ensemble_config.json')) as f:
    cfg = json.load(f)
expert_cfgs = cfg['expert_configs']
num_stocks = cfg['num_stocks']
stockid2idx = cfg['stockid2idx']
feature_list = cfg['feature_list']
seq_len = cfg['sequence_length']
feat_num = cfg['feature_num']
input_dim = cfg['input_dim']

print(f'特征: {input_dim}, 股票: {num_stocks}, 序列: {seq_len}')

# 数据
raw = pd.read_csv(DATA_FILE, dtype={'股票代码': str})
raw['日期'] = pd.to_datetime(raw['日期'])
latest = raw['日期'].max()
print(f'最新日期: {latest.date()}')

# 特征工程
fe = feature_engineer_func_map[feat_num]
fc = feature_cloums_map[feat_num]

df = raw.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= seq_len + 10]
print(f'有效股票: {len(groups)}')

with mp.Pool(min(8, mp.cpu_count())) as pool:
    plist = list(tqdm(pool.imap(fe, groups), total=len(groups), desc='特征工程'))
p = pd.concat(plist).reset_index(drop=True)
p['instrument'] = p['股票代码'].map(stockid2idx)
p = p.dropna(subset=['instrument'])
p['instrument'] = p['instrument'].astype(np.int64)
p['日期'] = pd.to_datetime(p['日期'])
p = _build_label_and_clean(p, drop_small_open=False)
p[fc] = p[fc].replace([np.inf, -np.inf], np.nan).fillna(0.0)

scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
p[fc] = scaler.transform(p[fc])

# 构建推理序列
stock_ids = sorted(raw['股票代码'].unique())
seqs, seq_ids = [], []
for sid in stock_ids:
    hist = p[(p['股票代码'] == sid) & (p['日期'] <= latest)].sort_values('日期').tail(seq_len)
    if len(hist) == seq_len:
        seqs.append(hist[fc].values.astype(np.float32))
        seq_ids.append(sid)

x = torch.from_numpy(np.asarray(seqs, dtype=np.float32)).unsqueeze(0).to(device)
print(f'推理股票: {len(seq_ids)}')

# V1专家列表
V1_NAMES = ['transformer_deep', 'transformer_wide', 'transformer_balanced',
            'transformer_attention', 'transformer_lite', 'month_seasonal']
v1_cfgs = [c for c in expert_cfgs if c['name'] in V1_NAMES]

# 加载专家
experts = []
for ecfg in v1_cfgs:
    name = ecfg['name']
    path = os.path.join(MODEL_DIR, f'expert_{name}.pth')
    t = ecfg.get('type', 'transformer')
    if t == 'transformer':
        m = StockTransformerExpert(input_dim, ecfg, num_stocks)
    elif t == 'month_seasonal':
        m = MonthSeasonalExpert(input_dim, ecfg, num_stocks)
    else:
        continue
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    experts.append(m)
    print(f'  {name} ✓')

# MC预测
print(f'MC推理 x20...')
all_scores = []
for e in experts:
    e.train()
    mc = []
    with torch.no_grad():
        for _ in range(20):
            mc.append(e(x).squeeze(0))
    all_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())

# V1 Meta
es = torch.from_numpy(np.stack(all_scores, axis=-1)).unsqueeze(0).float().to(device)
meta = MetaAggregator(len(experts), num_stocks).to(device)
meta.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'meta_aggregator.pth'), map_location=device))
meta.eval()
with torch.no_grad():
    final = meta(es).squeeze(0).cpu().numpy()

# Top5等权
order = np.argsort(final)[::-1]
top5 = [seq_ids[i] for i in order[:5]]

# 保存
os.makedirs('../output/', exist_ok=True)
pd.DataFrame({'stock_id': top5, 'weight': [0.2] * 5}).to_csv(OUTPUT_FILE, index=False)
print(f'\nV1本地预测 Top5: {top5}')
print(f'已保存: {OUTPUT_FILE}')
