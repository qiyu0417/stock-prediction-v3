"""Smart ensemble: consensus filtering + conditional switching"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda'); set_seed(42)

# --- data ---
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str}); train_df['股票代码']=train_df['股票代码'].astype(str).str.zfill(6); train_df['日期']=pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str}); test_df['股票代码']=test_df['股票代码'].astype(str).str.zfill(6); test_df['日期']=pd.to_datetime(test_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=['股票代码', '日期'], keep='last')
test_dates = sorted(test_df['日期'].unique()); raw_data = full_df.copy()
all_sids = sorted(full_df['股票代码'].unique()); sid2idx = {s:i for i,s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]; fcols_all = feature_cloums_map[FEATURE_NUM]

base_dir = 'model/stock_emb_8_hybrid'
with open(os.path.join(base_dir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
fd_full = cfg['feature_dim']; ns = cfg['num_stocks']
scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
with open(os.path.join(base_dir, 'winsor_bounds.json'), 'r') as f: wb = json.load(f)

df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed['instrument'] = processed['股票代码'].map(sid2idx); processed = processed.dropna(subset=['instrument']).copy()
processed['instrument'] = processed['instrument'].astype(np.int64)
processed = _build_label_and_clean(processed, drop_small_open=True)
processed[fcols_all] = processed[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
for col, (lo, hi) in wb.items():
    if col in processed.columns: processed[col] = processed[col].clip(lo, hi)
processed[fcols_all] = scaler.transform(processed[fcols_all])

alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]
nf_alpha = len(alpha_f)

import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__; _orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

def load_models(mdir, fd, loader_type='full'):
    _em.StockTransformerExpert.__init__ = _orig_ti; _em.ConvStockExpert.__init__ = _orig_ci
    with open(os.path.join(mdir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8); models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(fd, e, ns) if ec['type']=='transformer' else ConvStockExpert(fd, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)
    return models

print('Loading models...')
H = load_models('model/stock_emb_8_hybrid', fd_full)
A = load_models('model/stock_emb_8_alpha158', nf_alpha)
E = load_models('model/stock_emb_8_ema', fd_full)

def mc_scores(models, seq_t):
    all_s = []
    for _ in range(MC):
        ps = []
        for m in models:
            with torch.no_grad():
                p = m(seq_t)
                if isinstance(p, tuple): p = p[0]
                ps.append(p[0].cpu().numpy())
        all_s.append(np.mean(ps, axis=0))
    return np.mean(all_s, axis=0)

def compute_week(pd_str, score_array, valid, sids, label=''):
    raw = {sid: float(score_array[i]) if valid[i] else -float('inf') for i, sid in enumerate(sids)}
    data = raw_data[raw_data['日期'] <= pd_str]
    filt = volatility_filter(data, sids, pd_str, top_pct=0.95)
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
        if len(r1)>0 and len(r5)>0: ret += (float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])*w
    return ret, [s for s,_ in picks]

print('\n=== Smart Ensemble Strategies ===')
results = {}

for pd_str in ['2026-06-01', '2026-06-08']:
    hist = processed[processed['日期'] <= pd_str]
    sids = sorted(hist['股票代码'].unique())
    n_stocks = len(sids)

    # Full feature seq
    seq_full = np.zeros((1, n_stocks, SEQ, fd_full), dtype=np.float32)
    valid = np.zeros(n_stocks, dtype=bool)
    for i, sid in enumerate(sids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ: seq_full[0, i] = sd[fcols_all].values[-SEQ:].astype(np.float32); valid[i] = True

    # Alpha seq
    seq_alpha = np.zeros((1, n_stocks, SEQ, nf_alpha), dtype=np.float32)
    for i, sid in enumerate(sids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ: seq_alpha[0, i] = sd[alpha_f].values[-SEQ:].astype(np.float32)

    seq_t_full = torch.FloatTensor(seq_full).to(device)
    seq_t_alpha = torch.FloatTensor(seq_alpha).to(device)

    sc_h = mc_scores(H, seq_t_full)
    sc_a = mc_scores(A, seq_t_alpha)
    sc_e = mc_scores(E, seq_t_full)

    # Market regime detection: look at recent market trend
    hist_market = hist[hist['日期'] >= pd.to_datetime(pd_str) - pd.Timedelta(days=10)]
    market_up_pct = (hist_market.groupby('日期')['涨跌幅'].mean() > 0).mean()
    trend_strength = hist_market.groupby('日期')['涨跌幅'].mean().mean() * 100

    print(f'\n{pd_str}: trend={trend_strength:.2f}% up_pct={market_up_pct:.1%}')

    # Standalone
    wr_h, p_h = compute_week(pd_str, sc_h, valid, sids)
    wr_a, p_a = compute_week(pd_str, sc_a, valid, sids)

    # Strategy 1: Simple average (baseline)
    sc_simple = (sc_h + sc_a) / 2.0
    wr1, p1 = compute_week(pd_str, sc_simple, valid, sids)

    # Strategy 2: Consensus filter - only keep stocks where both agree on top-10
    def consensus_filter(sc_a, sc_h, sc_ids):
        top_a = set(np.argsort(sc_a)[-10:])
        top_h = set(np.argsort(sc_h)[-10:])
        consensus = top_a & top_h
        sc_combined = sc_h.copy()
        boost_idx = list(consensus)
        if boost_idx:
            sc_combined[boost_idx] = sc_h[boost_idx] * 1.15  # boost consensus stocks
        return sc_combined

    sc_consensus = consensus_filter(sc_a, sc_h, sids)
    wr2, p2 = compute_week(pd_str, sc_consensus, valid, sids)

    # Strategy 3: Weighted by market trend (strong trend = more Alpha, weak = more Hybrid)
    w = np.clip(market_up_pct, 0.3, 0.7)  # cap alpha weight between 30-70%
    sc_weighted = w * sc_a + (1 - w) * sc_h
    wr3, p3 = compute_week(pd_str, sc_weighted, valid, sids)

    # Strategy 4: EMA + Hybrid + Alpha158 3-model with consensus
    sc_triple = (sc_h + sc_e + sc_consensus) / 3.0
    wr4, p4 = compute_week(pd_str, sc_triple, valid, sids)

    print(f'  Hybrid:     {p_h} | {wr_h*100:+.2f}%')
    print(f'  Alpha158:   {p_a} | {wr_a*100:+.2f}%')
    print(f'  Simple avg: {p1} | {wr1*100:+.2f}%')
    print(f'  Consensus:  {p2} | {wr2*100:+.2f}%')
    print(f'  Weighted:   {p3} | {wr3*100:+.2f}%')

print('\nDone!')
