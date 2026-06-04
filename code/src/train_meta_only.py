"""仅训练元调度器 - 加载已有专家，收集预测，训练meta"""
import sys, os, json
sys.path.insert(0, 'code/src')

import torch, torch.nn.functional as F
import numpy as np, pandas as pd, joblib
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, WeightedRankingLoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)
from ensemble_config import *
from ensemble_models import *
from utils import create_ranking_dataset_vectorized

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 加载数据
full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
stockid2idx = {s: i for i, s in enumerate(sorted(full_df['股票代码'].unique()))}
num_stocks = len(stockid2idx)

# 用ensemble_config中的配置
# input_dim = 特征列总数(含instrument)
input_dim = len(feature_cloums_map[FEATURE_NUM])
expert_configs = EXPERT_CONFIGS
seq_len = SEQUENCE_LENGTH

# 加载所有专家
print("加载专家模型...")
experts = []
for cfg in expert_configs:
    name = cfg['name']
    path = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
    if not os.path.exists(path):
        print(f"  跳过 {name}: 未找到")
        continue
    t = cfg.get('type', 'transformer')
    if t == 'transformer':
        m = StockTransformerExpert(input_dim, cfg, num_stocks)
    elif t == 'month_seasonal':
        m = MonthSeasonalExpert(input_dim, cfg, num_stocks)
    else:
        continue
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    experts.append(m)
    print(f"  {name} ✓")

print(f"共 {len(experts)} 个专家")

# 特征工程（全量数据用于元调度器）
print("\n特征工程...")
fe = feature_engineer_func_map[FEATURE_NUM]
fc = feature_cloums_map[FEATURE_NUM]

df = full_df.copy()
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= seq_len + 10]

import multiprocessing as mp
with mp.Pool(min(8, mp.cpu_count())) as pool:
    plist = list(tqdm(pool.imap(fe, groups), total=len(groups), desc='特征工程'))

p = pd.concat(plist).reset_index(drop=True)
p['instrument'] = p['股票代码'].map(stockid2idx)
p = p.dropna(subset=['instrument'])
p['instrument'] = p['instrument'].astype(np.int64)
p = _build_label_and_clean(p, drop_small_open=True)
p[fc] = p[fc].replace([np.inf, -np.inf], np.nan).dropna(subset=fc)
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
p[fc] = scaler.fit_transform(p[fc])

# 构建排序数据集
train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(p, fc, seq_len)
print(f"训练样本: {len(train_seq)}")

train_ds = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

# 收集各专家预测
print("收集专家预测...")
all_scores, all_tgt, all_mask = [], [], []
with torch.no_grad():
    for batch in tqdm(train_dl, desc='专家推理'):
        seq = batch['sequences'].to(device)
        batch_scores = []
        for e in experts:
            e.eval()
            batch_scores.append(e.predict_with_mc_dropout(seq, num_samples=5))
        all_scores.append(torch.stack(batch_scores, dim=-1))
        all_tgt.append(batch['targets'])
        all_mask.append(batch['masks'])

# 训练元调度器
print(f"\n训练元调度器 ({META_EPOCHS} epochs)...")
meta = MetaAggregator(len(experts), num_stocks, hidden_dim=META_HIDDEN_DIM).to(device)
opt = torch.optim.Adam(meta.parameters(), lr=META_LR)
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=META_EPOCHS//3, T_mult=2, eta_min=META_LR*0.01)

for epoch in range(META_EPOCHS):
    meta.train()
    total_loss = 0
    nb = 0
    for i in range(len(all_scores)):
        es = all_scores[i].to(device)
        tg = all_tgt[i].to(device)
        mk = all_mask[i].to(device)

        opt.zero_grad()
        final = meta(es)
        mf = final * mk + (1 - mk) * (-1e9)

        loss = None
        B = es.size(0)
        for j in range(B):
            vi = mk[j].nonzero().squeeze()
            if vi.numel() <= 1: continue
            if vi.dim() == 0: vi = vi.unsqueeze(0)
            vp = mf[j][vi]
            vt = tg[j][vi]
            _, si = torch.sort(vt, descending=True)
            r = torch.zeros_like(vt)
            r[si] = torch.arange(len(vt), 0, -1, device=device, dtype=torch.float32)
            l = F.mse_loss(vp, r)
            loss = loss + l if loss is not None else l

        if loss is not None:
            (loss / B).backward()
            opt.step()
            total_loss += loss.item() / B
            nb += 1
    sch.step()
    if (epoch + 1) % 5 == 0:
        print(f"  [Meta] E{epoch+1:2d} Loss: {total_loss/max(nb,1):.4f}")

torch.save(meta.state_dict(), os.path.join(OUTPUT_DIR, 'meta_aggregator.pth'))
print("元调度器已保存!")
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print("全部完成!")
