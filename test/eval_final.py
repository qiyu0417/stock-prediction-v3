"""Evaluate individual models: EMA, GLU Rank, RankGLU"""
import sys, os, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC_PASSES = 5
SEQUENCE_LENGTH = 60
device = torch.device('cuda')
set_seed(42)

train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6)
train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.drop_duplicates(subset=['股票代码', '日期'], keep='last')
test_dates = sorted(test_df['日期'].unique())
raw_data = full_df.copy()
all_stock_ids = sorted(full_df['股票代码'].unique())
stockid2idx = {s: i for i, s in enumerate(all_stock_ids)}

from config_stock_emb_8 import FEATURE_NUM
feature_engineer = feature_engineer_func_map[FEATURE_NUM]
feature_columns = feature_cloums_map[FEATURE_NUM]

# GatedFeatureAttention patch for RankGLU
class GatedFA(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1), nn.Softmax(dim=1))
        self.gate = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, d_model), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_weights = self.attention(x)
        attended = torch.sum(x * attn_weights, dim=1)
        return self.dropout(attended * self.gate(attended))


def quick_eval(model_dir, label, patch_cls=None):
    print(f'\n=== {label} ===')
    with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
        cfg = json.load(f)
    feature_dim = cfg['feature_dim']
    num_stocks = cfg['num_stocks']
    embed_dim = cfg.get('stock_embed_dim', 8)
    expert_configs = cfg['expert_configs']

    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    with open(os.path.join(model_dir, 'winsor_bounds.json'), 'r') as f:
        winsor_bounds = json.load(f)

    df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQUENCE_LENGTH + 10]
    processed = pd.concat([feature_engineer(g) for g in groups]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=True)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=feature_columns)
    for col, (lo, hi) in winsor_bounds.items():
        if col in processed.columns:
            processed[col] = processed[col].clip(lo, hi)
    processed[feature_columns] = scaler.transform(processed[feature_columns])
    n_feats = len(feature_columns)

    import ensemble_models as _em
    _orig_fa = _em.FeatureAttention
    if patch_cls:
        _em.FeatureAttention = patch_cls

    from ensemble_models import StockTransformerExpert, ConvStockExpert
    models = []
    for ec in expert_configs:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        ec_copy = dict(ec)
        ec_copy['stock_embed_dim'] = embed_dim
        if ec['type'] == 'transformer':
            model = StockTransformerExpert(feature_dim, ec_copy, num_stocks)
        else:
            model = ConvStockExpert(feature_dim, ec_copy, num_stocks)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.train()
        models.append(model)
    _em.FeatureAttention = _orig_fa

    print(f'  Experts: {len(models)}')
    if not models:
        return None

    pred_dates = ['2026-06-01', '2026-06-08']
    rets = []
    for pred_date_str in pred_dates:
        hist = processed[processed['日期'] <= pred_date_str]
        avail_stocks = hist['股票代码'].unique()
        stock_ids = sorted(avail_stocks)
        n_stocks = len(stock_ids)
        if n_stocks < 5:
            continue

        sequences = np.zeros((1, n_stocks, SEQUENCE_LENGTH, n_feats), dtype=np.float32)
        valid_mask = np.zeros(n_stocks, dtype=bool)
        for i, sid in enumerate(stock_ids):
            sd = hist[hist['股票代码'] == sid].sort_values('日期')
            if len(sd) >= SEQUENCE_LENGTH:
                sequences[0, i] = sd[feature_columns].values[-SEQUENCE_LENGTH:].astype(np.float32)
                valid_mask[i] = True

        seq_t = torch.FloatTensor(sequences).to(device)
        all_scores = []
        for _ in range(MC_PASSES):
            pass_scores = []
            for model in models:
                with torch.no_grad():
                    pred = model(seq_t)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    pass_scores.append(pred[0].cpu().numpy())
            all_scores.append(np.mean(pass_scores, axis=0))
        mc_scores = np.mean(all_scores, axis=0)

        raw_scores = {}
        for i, sid in enumerate(stock_ids):
            raw_scores[sid] = float(mc_scores[i]) if valid_mask[i] else -float('inf')

        data = raw_data[raw_data['日期'] <= pred_date_str]
        filtered_ids = volatility_filter(data, stock_ids, pred_date_str, top_pct=0.95)
        bounce_flags = bounce_confirm(data, filtered_ids, pred_date_str)
        quality_scores = compute_quality_score(data, filtered_ids, pred_date_str)

        final_scores = {}
        for sid in filtered_ids:
            score = raw_scores.get(sid, -float('inf'))
            if sid not in bounce_flags:
                score *= 0.92
            quality = quality_scores.get(sid, 0.5)
            score += (quality - 0.5) * 0.05
            final_scores[sid] = score

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top5_ids = [sid for sid, _ in ranked[:5]]
        selected, weights = equal_weight_allocate(top5_ids)
        picks = list(zip(selected, weights))

        t1_date = None
        for d in test_dates:
            if d >= pd.to_datetime(pred_date_str):
                t1_date = d
                break
        t5_range = [d for d in test_dates if d >= pd.to_datetime(pred_date_str)]
        t5_date = t5_range[min(4, len(t5_range) - 1)]

        week_rets = []
        for sid, weight in picks:
            t1 = test_df[(test_df['股票代码'] == sid) & (test_df['日期'] == t1_date)]
            t5 = test_df[(test_df['股票代码'] == sid) & (test_df['日期'] == t5_date)]
            if len(t1) == 0 or len(t5) == 0:
                r = 0.0
            else:
                r = (float(t5.iloc[0]['开盘']) - float(t1.iloc[0]['开盘'])) / float(t1.iloc[0]['开盘'])
            week_rets.append(r * weight)
        wr = sum(week_rets)
        sids = [p[0] for p in picks]
        rets.append(wr)
        print(f'  {pred_date_str}: {sids} | {wr*100:+.2f}%')

    avg = np.mean(rets)
    w1 = rets[0] * 100
    w2 = rets[1] * 100
    print(f'  => W1: {w1:+.2f}%  W2: {w2:+.2f}%  Avg: {avg*100:+.2f}%')
    del models
    gc.collect()
    torch.cuda.empty_cache()
    return {'w1': w1, 'w2': w2, 'avg': avg * 100}


r_ema = quick_eval('model/stock_emb_8_ema', 'EMA')

# GLU Rank needs patched ranking_layers
import ensemble_models as _em2
class GatedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.value = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        v = self.value(x)
        g = torch.sigmoid(self.gate(x))
        return self.dropout(self.norm(v * g))

_orig_trans_init = _em2.StockTransformerExpert.__init__
_orig_conv_init = _em2.ConvStockExpert.__init__
def _patched_trans_init(self, input_dim, expert_config, num_stocks):
    _orig_trans_init(self, input_dim, expert_config, num_stocks)
    dm = self.d_model
    do = expert_config.get('dropout', 0.1)
    self.ranking_layers = nn.Sequential(
        GatedLinear(dm, dm, do), GatedLinear(dm, dm//2, do),
        nn.LayerNorm(dm//2), nn.ReLU(), nn.Dropout(do))
def _patched_conv_init(self, input_dim, expert_config, num_stocks):
    _orig_conv_init(self, input_dim, expert_config, num_stocks)
    h = self.d_model
    do = expert_config.get('dropout', 0.1)
    self.ranking_layers = nn.Sequential(
        GatedLinear(h, h, do), GatedLinear(h, h//2, do),
        nn.LayerNorm(h//2), nn.ReLU(), nn.Dropout(do))
_em2.StockTransformerExpert.__init__ = _patched_trans_init
_em2.ConvStockExpert.__init__ = _patched_conv_init

r_glurank = quick_eval('model/stock_emb_8_glurank', 'GLU Rank')

# Restore
_em2.StockTransformerExpert.__init__ = _orig_trans_init
_em2.ConvStockExpert.__init__ = _orig_conv_init
r_rankglu = quick_eval('model/stock_emb_8_rankglu', 'RankGLU', GatedFA)

print('\n' + '=' * 70)
print(f'{"Model":<25} {"Jun W1":>10} {"Jun W2":>10} {"Avg":>10}')
print('-' * 55)
ref = {
    'EMA': r_ema,
    'GLU Rank': r_glurank,
    'RankGLU': r_rankglu,
    'Hybrid dim=8': {'w1': 2.93, 'w2': 13.55, 'avg': 8.24},
    'k=3 T=0.5': {'w1': 2.50, 'w2': 8.83, 'avg': 5.67},
    'Multitask': {'w1': 1.16, 'w2': 16.58, 'avg': 8.87},
    'TopK-Dropout': {'w1': 5.16, 'w2': 5.02, 'avg': 5.09},
}
for name in ['EMA', 'GLU Rank', 'RankGLU', 'Hybrid dim=8', 'k=3 T=0.5', 'Multitask', 'TopK-Dropout']:
    r = ref.get(name)
    if r:
        print(f'{name:<25} {r["w1"]:>+9.2f}% {r["w2"]:>+9.2f}% {r["avg"]:>+9.2f}%')

print('\n--- Ensemble ---')
ensembles = {
    'EMA + Hybrid': {'w1': 4.80, 'w2': 13.85, 'avg': 9.33},
    'Hybrid + k=3': {'w1': 3.82, 'w2': 13.85, 'avg': 8.83},
    'Multitask + Hybrid + k3': {'w1': 2.06, 'w2': 17.37, 'avg': 9.71},
    'RankGLU + Hybrid + k3': {'w1': 4.62, 'w2': 13.55, 'avg': 9.09},
    'EMA + Hybrid + k3': {'w1': 3.01, 'w2': 11.70, 'avg': 7.35},
}
for name, r in ensembles.items():
    print(f'{name:<30} {r["w1"]:>+9.2f}% {r["w2"]:>+9.2f}% {r["avg"]:>+9.2f}%')

print('\nDone!')
