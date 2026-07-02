"""MC=20 stable evaluation of key models"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC_PASSES = 20; SEQ = 60
device = torch.device('cuda')
set_seed(42)

# --- data ---
print("Loading data...")
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str}); train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6); train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str}); test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6); test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=['股票代码', '日期'], keep='last')
test_dates = sorted(test_df['日期'].unique()); raw_data = full_df.copy()
all_stock_ids = sorted(full_df['股票代码'].unique()); stockid2idx = {s: i for i, s in enumerate(all_stock_ids)}

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]; fcols = feature_cloums_map[FEATURE_NUM]

base_dir = 'model/stock_emb_8_hybrid'
with open(os.path.join(base_dir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
fd = cfg['feature_dim']; ns = cfg['num_stocks']
scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
with open(os.path.join(base_dir, 'winsor_bounds.json'), 'r') as f: wb = json.load(f)

df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed['instrument'] = processed['股票代码'].map(stockid2idx); processed = processed.dropna(subset=['instrument']).copy()
processed['instrument'] = processed['instrument'].astype(np.int64)
processed = _build_label_and_clean(processed, drop_small_open=True)
processed[fcols] = processed[fcols].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols)
for col, (lo, hi) in wb.items():
    if col in processed.columns: processed[col] = processed[col].clip(lo, hi)
processed[fcols] = scaler.transform(processed[fcols]); nf = len(fcols)

import ensemble_models as _em
_orig_trans_init = _em.StockTransformerExpert.__init__
_orig_conv_init = _em.ConvStockExpert.__init__
_orig_fa = _em.FeatureAttention
from ensemble_models import StockTransformerExpert, ConvStockExpert

def load_std(mdir):
    _em.FeatureAttention = _orig_fa
    _em.StockTransformerExpert.__init__ = _orig_trans_init
    _em.ConvStockExpert.__init__ = _orig_conv_init
    with open(os.path.join(mdir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(fd, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(fd, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)
    return models

def load_se(mdir):
    _em.FeatureAttention = _orig_fa
    _em.StockTransformerExpert.__init__ = _orig_trans_init
    _em.ConvStockExpert.__init__ = _orig_conv_init
    class SEBlock(nn.Module):
        def __init__(self, c, r=4): super().__init__(); self.fc = nn.Sequential(nn.Linear(c, c//r), nn.ReLU(), nn.Linear(c//r, c), nn.Sigmoid())
        def forward(self, x): return x * self.fc(x)
    class SEWrapper(nn.Module):
        def __init__(self, fa, d): super().__init__(); self.fa = fa; self.se = SEBlock(d)
        def forward(self, x): return self.se(self.fa(x))
    def _se_t(self, i, ec, n): _orig_trans_init(self, i, ec, n); self.feature_attention = SEWrapper(self.feature_attention, self.d_model)
    def _se_c(self, i, ec, n): _orig_conv_init(self, i, ec, n); self.feature_attention = SEWrapper(self.feature_attention, self.d_model)
    _em.StockTransformerExpert.__init__ = _se_t; _em.ConvStockExpert.__init__ = _se_c
    with open(os.path.join(mdir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(fd, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(fd, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)
    _em.StockTransformerExpert.__init__ = _orig_trans_init; _em.ConvStockExpert.__init__ = _orig_conv_init
    return models

def load_rglu(mdir):
    _em.StockTransformerExpert.__init__ = _orig_trans_init; _em.ConvStockExpert.__init__ = _orig_conv_init
    class _GatedFA(nn.Module):
        def __init__(self, d, dropout=0.1):
            super().__init__()
            self.attention = nn.Sequential(nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1), nn.Softmax(dim=1))
            self.gate = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, d), nn.Sigmoid())
            self.dropout = nn.Dropout(dropout)
        def forward(self, x): attn = self.attention(x); attd = torch.sum(x * attn, dim=1); return self.dropout(attd * self.gate(attd))
    _em.FeatureAttention = _GatedFA
    with open(os.path.join(mdir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(fd, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(fd, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)
    _em.FeatureAttention = _orig_fa
    return models

def compute_week(pd_str, score_array, valid_mask, stock_ids):
    raw = {sid: float(score_array[i]) if valid_mask[i] else -float('inf') for i, sid in enumerate(stock_ids)}
    data = raw_data[raw_data['日期'] <= pd_str]
    filt = volatility_filter(data, stock_ids, pd_str, top_pct=0.95)
    bnc = bounce_confirm(data, filt, pd_str)
    qual = compute_quality_score(data, filt, pd_str)
    final = {}
    for sid in filt:
        s = raw.get(sid, -float('inf'))
        if sid not in bnc: s *= 0.92
        s += (qual.get(sid, 0.5) - 0.5) * 0.05; final[sid] = s
    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
    picks = list(zip(*equal_weight_allocate([s for s,_ in ranked])))
    t1 = next(d for d in test_dates if d >= pd.to_datetime(pd_str))
    t5r = [d for d in test_dates if d >= pd.to_datetime(pd_str)]; t5 = t5r[min(4, len(t5r)-1)]
    ret = 0.0
    for sid, w in picks:
        r1 = test_df[(test_df['股票代码']==sid)&(test_df['日期']==t1)]
        r5 = test_df[(test_df['股票代码']==sid)&(test_df['日期']==t5)]
        if len(r1)>0 and len(r5)>0:
            ret += (float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])*w
    return ret, [s for s,_ in picks]

def eval_model(models, label):
    rets = []
    for pd_str in ['2026-06-01', '2026-06-08']:
        hist = processed[processed['日期'] <= pd_str]
        sids = sorted(hist['股票代码'].unique())
        seq = np.zeros((1, len(sids), SEQ, nf), dtype=np.float32); valid = np.zeros(len(sids), dtype=bool)
        for i, sid in enumerate(sids):
            sd = hist[hist['股票代码'] == sid].sort_values('日期')
            if len(sd) >= SEQ: seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32); valid[i] = True
        seq_t = torch.FloatTensor(seq).to(device)
        all_s = []
        for _ in range(MC_PASSES):
            ps = []
            for m in models:
                with torch.no_grad():
                    p = m(seq_t)
                    if isinstance(p, tuple): p = p[0]
                    ps.append(p[0].cpu().numpy())
            all_s.append(np.mean(ps, axis=0))
        mc = np.mean(all_s, axis=0)
        wr, picks = compute_week(pd_str, mc, valid, sids)
        rets.append(wr)
        print(f'  [{label}] {pd_str}: {picks} | {wr*100:+.2f}%')
    avg = np.mean(rets)
    print(f'  => W1: {rets[0]*100:+.2f}%  W2: {rets[1]*100:+.2f}%  Avg: {avg*100:+.2f}%')
    return {'w1': rets[0]*100, 'w2': rets[1]*100, 'avg': avg*100}

results = {}
for name, loader_fn, mdir in [
    ('Hybrid', load_std, 'model/stock_emb_8_hybrid'),
    ('EMA', load_std, 'model/stock_emb_8_ema'),
    ('SE v2', load_se, 'model/stock_emb_8_se_v2'),
    ('RankGLU', load_rglu, 'model/stock_emb_8_rankglu'),
]:
    print(f'\nLoading {name}...')
    models = loader_fn(mdir)
    print(f'  {len(models)} experts')
    results[name] = eval_model(models, name)

# Quick ensemble: EMA + Hybrid
print(f'\n--- Ensemble EMA + Hybrid ---')
ema_m = load_std('model/stock_emb_8_ema')
hyb_m = load_std('model/stock_emb_8_hybrid')

for pd_str in ['2026-06-01', '2026-06-08']:
    hist = processed[processed['日期'] <= pd_str]
    sids = sorted(hist['股票代码'].unique())
    seq = np.zeros((1, len(sids), SEQ, nf), dtype=np.float32); valid = np.zeros(len(sids), dtype=bool)
    for i, sid in enumerate(sids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ: seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32); valid[i] = True
    seq_t = torch.FloatTensor(seq).to(device)
    all_model_scores = []
    for models in [ema_m, hyb_m]:
        mc_list = []
        for _ in range(MC_PASSES):
            ps = []
            for m in models:
                with torch.no_grad():
                    p = m(seq_t)
                    if isinstance(p, tuple): p = p[0]
                    ps.append(p[0].cpu().numpy())
            mc_list.append(np.mean(ps, axis=0))
        all_model_scores.append(np.mean(mc_list, axis=0))
    mc = np.mean(all_model_scores, axis=0)
    wr, picks = compute_week(pd_str, mc, valid, sids)
    print(f'  [EMA + Hybrid] {pd_str}: {picks} | {wr*100:+.2f}%')

print(f'\n{"="*50}')
print(f'{"Model":<20} {"Jun W1":>10} {"Jun W2":>10} {"Avg":>10}')
print('-'*45)
for name, r in sorted(results.items(), key=lambda x: x[1]['avg'], reverse=True):
    print(f'{name:<20} {r["w1"]:>+9.2f}% {r["w2"]:>+9.2f}% {r["avg"]:>+9.2f}%')
print('Done! MC=20 stable.')
