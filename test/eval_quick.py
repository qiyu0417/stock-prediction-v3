"""Quick focused eval: SE v2 + top ensembles only"""
import os, sys, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC_PASSES = 5
SEQUENCE_LENGTH = 60
device = torch.device('cuda')
set_seed(42)

# ---- model loaders ----
import ensemble_models as _em
_orig_trans_init = _em.StockTransformerExpert.__init__
_orig_conv_init = _em.ConvStockExpert.__init__
_orig_fa = _em.FeatureAttention

def load_rankglu(model_dir, feature_dim, num_stocks):
    _em.StockTransformerExpert.__init__ = _orig_trans_init
    _em.ConvStockExpert.__init__ = _orig_conv_init
    class _GatedFeatureAttention(nn.Module):
        def __init__(self, d_model, dropout=0.1):
            super().__init__()
            self.attention = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1), nn.Softmax(dim=1))
            self.gate = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, d_model), nn.Sigmoid())
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            attn_weights = self.attention(x)
            attended = torch.sum(x * attn_weights, dim=1)
            return self.dropout(attended * self.gate(attended))
    _em.FeatureAttention = _GatedFeatureAttention
    from ensemble_models import StockTransformerExpert, ConvStockExpert
    with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
        cfg = json.load(f)
    embed_dim = cfg.get('stock_embed_dim', 8)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        ec_copy = dict(ec); ec_copy['stock_embed_dim'] = embed_dim
        m = StockTransformerExpert(feature_dim, ec_copy, num_stocks) if ec['type'] == 'transformer' else ConvStockExpert(feature_dim, ec_copy, num_stocks)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.to(device); m.train()
        models.append(m)
    _em.FeatureAttention = _orig_fa
    return models

def load_standard(model_dir, feature_dim, num_stocks):
    _em.FeatureAttention = _orig_fa
    _em.StockTransformerExpert.__init__ = _orig_trans_init
    _em.ConvStockExpert.__init__ = _orig_conv_init
    from ensemble_models import StockTransformerExpert, ConvStockExpert
    with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
        cfg = json.load(f)
    embed_dim = cfg.get('stock_embed_dim', 8)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        ec_copy = dict(ec); ec_copy['stock_embed_dim'] = embed_dim
        m = StockTransformerExpert(feature_dim, ec_copy, num_stocks) if ec['type'] == 'transformer' else ConvStockExpert(feature_dim, ec_copy, num_stocks)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.to(device); m.train()
        models.append(m)
    return models

def load_se_v2(model_dir, feature_dim, num_stocks):
    _em.FeatureAttention = _orig_fa
    _em.StockTransformerExpert.__init__ = _orig_trans_init
    _em.ConvStockExpert.__init__ = _orig_conv_init
    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(channels, channels//4), nn.ReLU(), nn.Linear(channels//4, channels), nn.Sigmoid())
        def forward(self, x): return x * self.fc(x)
    class SEWrapper(nn.Module):
        def __init__(self, fa_module, d_model):
            super().__init__()
            self.fa = fa_module; self.se = SEBlock(d_model)
        def forward(self, x): return self.se(self.fa(x))
    def _se_trans_init(self, input_dim, expert_config, num_stocks):
        _orig_trans_init(self, input_dim, expert_config, num_stocks)
        self.feature_attention = SEWrapper(self.feature_attention, self.d_model)
    def _se_conv_init(self, input_dim, expert_config, num_stocks):
        _orig_conv_init(self, input_dim, expert_config, num_stocks)
        self.feature_attention = SEWrapper(self.feature_attention, self.d_model)
    _em.StockTransformerExpert.__init__ = _se_trans_init
    _em.ConvStockExpert.__init__ = _se_conv_init
    from ensemble_models import StockTransformerExpert, ConvStockExpert
    with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
        cfg = json.load(f)
    embed_dim = cfg.get('stock_embed_dim', 8)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        ec_copy = dict(ec); ec_copy['stock_embed_dim'] = embed_dim
        m = StockTransformerExpert(feature_dim, ec_copy, num_stocks) if ec['type'] == 'transformer' else ConvStockExpert(feature_dim, ec_copy, num_stocks)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.to(device); m.train()
        models.append(m)
    _em.StockTransformerExpert.__init__ = _orig_trans_init
    _em.ConvStockExpert.__init__ = _orig_conv_init
    return models

# ---- data loading ----
print("Loading data...")
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6)
train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=['股票代码', '日期'], keep='last')
test_dates = sorted(test_df['日期'].unique())
raw_data = full_df.copy()
all_stock_ids = sorted(full_df['股票代码'].unique())
stockid2idx = {s: i for i, s in enumerate(all_stock_ids)}

from config_stock_emb_8 import FEATURE_NUM
feature_engineer = feature_engineer_func_map[FEATURE_NUM]
feature_columns = feature_cloums_map[FEATURE_NUM]

base_dir = 'model/stock_emb_8_hybrid'
with open(os.path.join(base_dir, 'ensemble_config.json'), 'r') as f:
    cfg = json.load(f)
feature_dim = cfg['feature_dim']; num_stocks = cfg['num_stocks']
scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
with open(os.path.join(base_dir, 'winsor_bounds.json'), 'r') as f:
    winsor_bounds = json.load(f)

df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQUENCE_LENGTH + 10]
processed = pd.concat([feature_engineer(g) for g in groups]).reset_index(drop=True)
processed['instrument'] = processed['股票代码'].map(stockid2idx)
processed = processed.dropna(subset=['instrument']).copy()
processed['instrument'] = processed['instrument'].astype(np.int64)
processed = _build_label_and_clean(processed, drop_small_open=True)
processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
for col, (lo, hi) in winsor_bounds.items():
    if col in processed.columns:
        processed[col] = processed[col].clip(lo, hi)
processed[feature_columns] = scaler.transform(processed[feature_columns])
n_feats = len(feature_columns)

# ---- eval helpers ----
def build_seq(pred_date_str):
    hist = processed[processed['日期'] <= pred_date_str]
    stock_ids = sorted(hist['股票代码'].unique())
    n_stocks = len(stock_ids)
    if n_stocks < 5: return None, None, None
    seq = np.zeros((1, n_stocks, SEQUENCE_LENGTH, n_feats), dtype=np.float32)
    valid = np.zeros(n_stocks, dtype=bool)
    for i, sid in enumerate(stock_ids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQUENCE_LENGTH:
            seq[0, i] = sd[feature_columns].values[-SEQUENCE_LENGTH:].astype(np.float32)
            valid[i] = True
    return torch.FloatTensor(seq).to(device), valid, stock_ids

def postprocess(scores, valid, stock_ids, pred_date_str):
    raw = {sid: float(scores[i]) if valid[i] else -float('inf') for i, sid in enumerate(stock_ids)}
    data = raw_data[raw_data['日期'] <= pred_date_str]
    filtered = volatility_filter(data, stock_ids, pred_date_str, top_pct=0.95)
    bounce = bounce_confirm(data, filtered, pred_date_str)
    quality = compute_quality_score(data, filtered, pred_date_str)
    final = {}
    for sid in filtered:
        s = raw.get(sid, -float('inf'))
        if sid not in bounce: s *= 0.92
        s += (quality.get(sid, 0.5) - 0.5) * 0.05
        final[sid] = s
    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
    picks = list(zip(*equal_weight_allocate([s for s,_ in ranked])))
    t1 = next(d for d in test_dates if d >= pd.to_datetime(pred_date_str))
    t5r = [d for d in test_dates if d >= pd.to_datetime(pred_date_str)]
    t5 = t5r[min(4, len(t5r)-1)]
    rets = []
    for sid, w in picks:
        r1 = test_df[(test_df['股票代码']==sid)&(test_df['日期']==t1)]
        r5 = test_df[(test_df['股票代码']==sid)&(test_df['日期']==t5)]
        r = 0.0 if len(r1)==0 or len(r5)==0 else (float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])
        rets.append(r*w)
    return sum(rets)

def eval_ensemble(model_list, label):
    rets = []
    for pd_str in ['2026-06-01', '2026-06-08']:
        seq_t, valid, stock_ids = build_seq(pd_str)
        if seq_t is None: continue
        all_scores = []
        for models in model_list:
            mc_scores = []
            for _ in range(MC_PASSES):
                ps = []
                for m in models:
                    with torch.no_grad():
                        p = m(seq_t)
                        if isinstance(p, tuple): p = p[0]
                        ps.append(p[0].cpu().numpy())
                mc_scores.append(np.mean(ps, axis=0))
            all_scores.append(np.mean(mc_scores, axis=0))
        ensemble = np.mean(all_scores, axis=0)
        wr = postprocess(ensemble, valid, stock_ids, pd_str)
        rets.append(wr)
        print(f'  [{label}] {pd_str}: {wr*100:+.2f}%')
    avg = np.mean(rets)
    print(f'  => W1: {rets[0]*100:+.2f}%  W2: {rets[1]*100:+.2f}%  Avg: {avg*100:+.2f}%')
    return {'w1': rets[0]*100, 'w2': rets[1]*100, 'avg': avg*100}

# ---- load models ----
print("Loading models...")
models = {}
for name, loader, mdir in [
    ('Hybrid', load_standard, 'model/stock_emb_8_hybrid'),
    ('EMA', load_standard, 'model/stock_emb_8_ema'),
    ('RankGLU', load_rankglu, 'model/stock_emb_8_rankglu'),
    ('EMA+ListMLE', load_standard, 'model/stock_emb_8_ema_listmle'),
    ('SE v2', load_se_v2, 'model/stock_emb_8_se_v2'),
]:
    print(f'  Loading {name}...')
    models[name] = loader(mdir, feature_dim, num_stocks)
    print(f'    {len(models[name])} experts')

# ---- eval ----
results = {}
print("\n=== STANDALONE ===")
for name, ml in models.items():
    results[name] = eval_ensemble([ml], name)

print("\n=== ENSEMBLES ===")
combos = [
    (['EMA', 'Hybrid'], 'EMA + Hybrid'),
    (['EMA', 'SE v2'], 'EMA + SE v2'),
    (['Hybrid', 'SE v2'], 'Hybrid + SE v2'),
    (['EMA', 'EMA+ListMLE'], 'EMA + EMA+ListMLE'),
    (['EMA', 'RankGLU'], 'EMA + RankGLU'),
    (['EMA', 'Hybrid', 'SE v2'], 'EMA + Hybrid + SE v2'),
]
for model_names, label in combos:
    if all(n in models for n in model_names):
        results[label] = eval_ensemble([models[n] for n in model_names], label)

print(f'\n{"="*60}')
print(f'{"Model":<30} {"Jun W1":>10} {"Jun W2":>10} {"Avg":>10}')
print('-'*50)
for name, r in sorted(results.items(), key=lambda x: x[1]['avg'], reverse=True):
    print(f'{name:<30} {r["w1"]:>+9.2f}% {r["w2"]:>+9.2f}% {r["avg"]:>+9.2f}%')
print('Done!')
