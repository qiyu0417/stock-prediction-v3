"""V1升级: 1)近期meta重训 2)专家历史得分加权 3)加入原始baseline"""
import sys, os, json, joblib, torch, numpy as np, pandas as pd
sys.path.insert(0, 'code/src')
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from ensemble_config import *
from ensemble_models import *
from model import StockTransformer as BaselineModel
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, WeightedRankingLoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ===== 专家历史得分（验证集final_score）=====
EXPERT_SCORES = {
    'transformer_deep': 0.7696,
    'transformer_wide': 0.6231,
    'transformer_balanced': 0.8515,
    'transformer_attention': 0.4392,
    'transformer_lite': 0.5093,
    'month_seasonal': 0.50,  # 保守估计
    'baseline': 0.35,  # 原始baseline权重较低
}

# ===== Baseline模型包装器 =====
class BaselineWrapper(nn.Module):
    """包装原始StockTransformer使其兼容集成接口"""
    def __init__(self, baseline_model, num_stocks):
        super().__init__()
        self.model = baseline_model
        self.num_stocks = num_stocks

    def forward(self, src):
        return self.model(src)

    def predict_with_mc_dropout(self, src, num_samples=10):
        # Baseline没有MC Dropout，多次前向结果相同
        self.eval()
        with torch.no_grad():
            scores = self.forward(src)
        return scores

# ===== 数据准备 =====
raw = pd.read_csv('data/train.csv', dtype={'股票代码': str})
raw['日期'] = pd.to_datetime(raw['日期'], format='mixed')
stockid2idx = {s: i for i, s in enumerate(sorted(raw['股票代码'].unique()))}
num_stocks = len(stockid2idx)
fe = feature_engineer_func_map[FEATURE_NUM]
fc = feature_cloums_map[FEATURE_NUM]
seq_len = SEQUENCE_LENGTH

def create_model(ecfg, input_dim):
    t = ecfg.get('type', 'transformer')
    if t == 'transformer': return StockTransformerExpert(input_dim, ecfg, num_stocks)
    if t == 'month_seasonal': return MonthSeasonalExpert(input_dim, ecfg, num_stocks)
    return None

def prepare_features(cutoff_date):
    """特征工程到指定日期"""
    train_df = raw[raw['日期'] <= cutoff_date].copy()
    df = train_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= seq_len + 10]
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        plist = list(tqdm(pool.imap(fe, groups), total=len(groups), desc='FE', leave=False))
    p = pd.concat(plist).reset_index(drop=True)
    p['instrument'] = p['股票代码'].map(stockid2idx)
    p = p.dropna(subset=['instrument'])
    p['instrument'] = p['instrument'].astype(np.int64)
    p['日期'] = pd.to_datetime(p['日期'], format='mixed')
    p = _build_label_and_clean(p, drop_small_open=True)
    p[fc] = p[fc].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return p, train_df

# ===== Step 1: 加载V1专家 + Baseline =====
print('\n=== Step 1: 加载专家 ===')

# 先做一次特征工程获取input_dim
p_full, _ = prepare_features(pd.Timestamp('2026-04-30'))
stock_ids_all = sorted(raw['股票代码'].unique())
latest_full = pd.Timestamp('2026-04-30')
seqs_full, seq_ids_full = [], []
for sid in stock_ids_all:
    hist = p_full[(p_full['股票代码'] == sid) & (p_full['日期'] <= latest_full)].sort_values('日期').tail(seq_len)
    if len(hist) == seq_len:
        seqs_full.append(hist[fc].values.astype(np.float32))
        seq_ids_full.append(sid)
input_dim = seqs_full[0].shape[-1]

V1_NAMES = ['transformer_deep', 'transformer_wide', 'transformer_balanced',
            'transformer_attention', 'transformer_lite', 'month_seasonal']
v1_cfgs = [c for c in EXPERT_CONFIGS if c['name'] in V1_NAMES]

experts = []
expert_names = []

# Load V1 experts
for ecfg in v1_cfgs:
    name = ecfg['name']
    path = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
    if not os.path.exists(path):
        path = os.path.join('model/v1_ensemble', f'expert_{name}.pth')
    if not os.path.exists(path): continue
    m = create_model(ecfg, input_dim)
    if m is None: continue
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    experts.append(m)
    expert_names.append(name)
    print(f'  {name} (历史得分: {EXPERT_SCORES.get(name, "?")})')

# Load baseline
baseline_path = 'model/60_158+39/best_model.pth'
baseline_scaler_path = 'model/60_158+39/scaler.pkl'
if os.path.exists(baseline_path):
    baseline_cfg = {
        'd_model': 256, 'nhead': 4, 'num_layers': 3,
        'dim_feedforward': 512, 'dropout': 0.1,
        'sequence_length': 60
    }
    baseline = BaselineModel(input_dim, baseline_cfg, num_stocks)
    baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    wrapped = BaselineWrapper(baseline, num_stocks).to(device)
    experts.append(wrapped)
    expert_names.append('baseline')
    print(f'  baseline (原始模型, 得分: {EXPERT_SCORES["baseline"]})')

n_experts = len(experts)
print(f'共 {n_experts} 个专家')

# ===== Step 2: 近期数据重训Meta =====
print('\n=== Step 2: 近期Meta重训（近6个月） ===')

meta_cutoff = pd.Timestamp('2026-03-06')
meta_start = meta_cutoff - pd.DateOffset(months=6)  # 用2025-10到2026-03的数据
print(f'Meta训练数据: {meta_start.date()} ~ {meta_cutoff.date()}')

# 特征工程（全量用于序列，但meta训练标签只用近期）
p_meta, train_meta = prepare_features(meta_cutoff)

# 构建排序数据集（全量）
from utils import create_ranking_dataset_vectorized
seqs, tgts, rels, stks = create_ranking_dataset_vectorized(p_meta, fc, seq_len)
print(f'排序样本: {len(seqs)}')

ds = RankingDataset(seqs, tgts, rels, stks)
dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

# 收集专家预测
print('收集专家预测...')
all_scores, all_tgt, all_mask = [], [], []
with torch.no_grad():
    for batch in tqdm(dl, desc='推理'):
        seq = batch['sequences'].to(device)
        batch_scores = []
        for e in experts:
            e.eval()
            if hasattr(e, 'predict_with_mc_dropout'):
                batch_scores.append(e.predict_with_mc_dropout(seq, num_samples=5))
            else:
                batch_scores.append(e(seq))
        all_scores.append(torch.stack(batch_scores, dim=-1))
        all_tgt.append(batch['targets'])
        all_mask.append(batch['masks'])

# 训练新Meta
meta = MetaAggregator(n_experts, num_stocks, hidden_dim=64).to(device)
opt = torch.optim.Adam(meta.parameters(), lr=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=7, T_mult=2, eta_min=1e-6)

for epoch in range(20):
    meta.train()
    tl = 0; nb = 0
    for i in range(len(all_scores)):
        es = all_scores[i].to(device); tg = all_tgt[i].to(device); mk = all_mask[i].to(device)
        opt.zero_grad()
        mf = meta(es) * mk + (1 - mk) * (-1e9)
        loss = None; B = es.size(0)
        for j in range(B):
            vi = mk[j].nonzero().squeeze()
            if vi.numel() <= 1: continue
            if vi.dim() == 0: vi = vi.unsqueeze(0)
            vp = mf[j][vi]; vt = tg[j][vi]
            _, si = torch.sort(vt, descending=True)
            r = torch.zeros_like(vt)
            r[si] = torch.arange(len(vt), 0, -1, device=device, dtype=torch.float32)
            l = F.mse_loss(vp, r)
            loss = loss + l if loss is not None else l
        if loss is not None:
            (loss / B).backward(); opt.step()
            tl += loss.item() / B; nb += 1
    sch.step()
    if (epoch + 1) % 5 == 0:
        print(f'  [Meta] E{epoch+1} Loss:{tl/max(nb,1):.4f}')

torch.save(meta.state_dict(), os.path.join(OUTPUT_DIR, 'meta_upgraded.pth'))
print('Meta已保存')

# ===== Step 3: 回测 =====
print('\n=== Step 3: 回测 ===')

# 用升级版scaler（V1的）
scaler = joblib.load(os.path.join('model/v1_ensemble', 'scaler.pkl'))

def predict_and_eval(name, cutoff, test_df):
    print(f'\n--- {name} ---')

    # 特征工程（用V1 scaler）
    p_eval, _ = prepare_features(cutoff)
    p_eval[fc] = scaler.transform(p_eval[fc])

    latest = cutoff
    stock_ids = sorted(raw[raw['日期'] <= cutoff]['股票代码'].unique())
    seqs_eval, seq_ids_eval = [], []
    for sid in stock_ids:
        hist = p_eval[(p_eval['股票代码'] == sid) & (p_eval['日期'] <= latest)].sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            seqs_eval.append(hist[fc].values.astype(np.float32))
            seq_ids_eval.append(sid)

    x = torch.from_numpy(np.asarray(seqs_eval, dtype=np.float32)).unsqueeze(0).to(device)
    print(f'股票: {len(seq_ids_eval)}')

    # MC预测
    all_scores = []
    for e in experts:
        e.eval()
        mc = []
        with torch.no_grad():
            for _ in range(20):
                if hasattr(e, 'predict_with_mc_dropout'):
                    mc.append(e.predict_with_mc_dropout(x, num_samples=1))
                else:
                    mc.append(e(x).squeeze(0))
        all_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())

    # 新Meta融合 (all_scores[i] already [1,N], stack->[1,N,E])
    es = torch.from_numpy(np.stack(all_scores, axis=-1)).float().to(device)
    with torch.no_grad():
        final = meta(es).squeeze(0).cpu().numpy()

    order = np.argsort(final)[::-1]
    top_ids = [seq_ids_eval[i] for i in order[:5]]

    # 等权（V1风格，简单有效）
    print(f'Top5 (等权):')
    for i in range(5):
        print(f'  {top_ids[i]}')

    # 评估
    if test_df is not None:
        td = test_df.copy()
        td['股票代码'] = td['股票代码'].astype(str)
        tf = td[td['股票代码'].isin(top_ids)].groupby('股票代码').tail(5)
        def cr(g):
            return (g.iloc[-1]['开盘'] - g.iloc[0]['开盘']) / g.iloc[0]['开盘']
        rets = tf.groupby('股票代码').apply(cr)
        total = sum(rets) * 0.2 if len(rets) > 0 else 0
        print(f'\n各股:')
        for sid, r in rets.items():
            print(f'  {sid}: {r:+.4%}')
        print(f'\n===== {name}: {total:.4%} =====')
        return total

    return top_ids

# Test set
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
s1 = predict_and_eval('V1+升级: 测试集', pd.Timestamp('2026-03-06'), test_df)

# May
may_df = raw[(raw['日期'] > pd.Timestamp('2026-04-30')) & (raw['日期'] <= pd.Timestamp('2026-05-10'))]
s2 = predict_and_eval('V1+升级: 5月', pd.Timestamp('2026-04-30'), may_df)

# June
print(f'\n--- V1+升级: 6月预测 ---')
top_june = predict_and_eval('6月', pd.Timestamp('2026-05-29'), None)
print(f'持仓: {top_june}')

print(f'\n===== 汇总 =====')
print(f'测试集: {s1:.4%}')
print(f'5月:    {s2:.4%}')
print(f'专家数: {n_experts} (6 V1 + baseline)')
