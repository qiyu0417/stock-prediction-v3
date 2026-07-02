"""Tech39 MC=20 eval — built on proven eval_smart_ensemble.py pattern"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS, _TECH_39_ONLY
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
tech_f = [f for f in _TECH_39_ONLY if f in fcols_all]
nf_alpha = len(alpha_f); nf_tech = len(tech_f)
print(f"Features: full={len(fcols_all)} alpha={nf_alpha} tech={nf_tech}")

import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__; _orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

def load_models(mdir, fd):
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
T = load_models('model/stock_emb_8_tech39', nf_tech)
print(f'Models: H={len(H)} A={len(A)} T={len(T)}')

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

print('\n=== Tech39 + 3-Model Ensemble (MC=20) ===')
results = {}

for pd_str in ['2026-06-01', '2026-06-08']:
    hist = processed[processed['日期'] <= pd_str]
    sids = sorted(hist['股票代码'].unique())
    n_stocks = len(sids)

    # Build all 3 sequence tensors from the SAME hist
    seq_full = np.zeros((1, n_stocks, SEQ, fd_full), dtype=np.float32)
    seq_alpha = np.zeros((1, n_stocks, SEQ, nf_alpha), dtype=np.float32)
    seq_tech = np.zeros((1, n_stocks, SEQ, nf_tech), dtype=np.float32)
    valid = np.zeros(n_stocks, dtype=bool)

    for i, sid in enumerate(sids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ:
            seq_full[0, i] = sd[fcols_all].values[-SEQ:].astype(np.float32)
            seq_alpha[0, i] = sd[alpha_f].values[-SEQ:].astype(np.float32)
            seq_tech[0, i] = sd[tech_f].values[-SEQ:].astype(np.float32)
            valid[i] = True

    sc_h = mc_scores(H, torch.FloatTensor(seq_full).to(device))
    sc_a = mc_scores(A, torch.FloatTensor(seq_alpha).to(device))
    sc_t = mc_scores(T, torch.FloatTensor(seq_tech).to(device))

    # Standalone
    wr_h, p_h = compute_week(pd_str, sc_h, valid, sids)
    wr_a, p_a = compute_week(pd_str, sc_a, valid, sids)
    wr_t, p_t = compute_week(pd_str, sc_t, valid, sids)

    # Market regime for weighting
    hist_market = hist[hist['日期'] >= pd.to_datetime(pd_str) - pd.Timedelta(days=10)]
    market_up_pct = (hist_market.groupby('日期')['涨跌幅'].mean() > 0).mean()

    print(f'\n{pd_str}: up_pct={market_up_pct:.1%}')
    print(f'  Hybrid:     {p_h} | {wr_h*100:+.2f}%')
    print(f'  Alpha158:   {p_a} | {wr_a*100:+.2f}%')
    print(f'  Tech39:     {p_t} | {wr_t*100:+.2f}%')

    # 2-model ensembles
    sc_ah = (sc_h + sc_a) / 2.0
    wr_ah, p_ah = compute_week(pd_str, sc_ah, valid, sids)
    sc_at = (sc_a + sc_t) / 2.0
    wr_at, p_at = compute_week(pd_str, sc_at, valid, sids)
    sc_ht = (sc_h + sc_t) / 2.0
    wr_ht, p_ht = compute_week(pd_str, sc_ht, valid, sids)

    # 3-model simple average
    sc_3avg = (sc_h + sc_a + sc_t) / 3.0
    wr_3avg, p_3avg = compute_week(pd_str, sc_3avg, valid, sids)

    # 3-model consensus (top-10 in at least 2 of 3)
    top_h = set(np.argsort(sc_h)[-10:])
    top_a = set(np.argsort(sc_a)[-10:])
    top_t = set(np.argsort(sc_t)[-10:])
    consensus_idx = [i for i in range(n_stocks) if sum([i in top_h, i in top_a, i in top_t]) >= 2]
    sc_cons3 = sc_h.copy()
    if consensus_idx:
        sc_cons3[consensus_idx] = sc_h[consensus_idx] * 1.10
    wr_cons3, p_cons3 = compute_week(pd_str, sc_cons3, valid, sids)

    # Weighted: Alpha(W1-best) + Hybrid(W2-best) + Tech(diversifier)
    w = np.clip(market_up_pct, 0.3, 0.7)
    sc_weighted = 0.4 * sc_h + 0.35 * sc_a + 0.25 * sc_t
    wr_w, p_w = compute_week(pd_str, sc_weighted, valid, sids)

    print(f'  A+H avg:     {p_ah} | {wr_ah*100:+.2f}%')
    print(f'  A+T avg:     {p_at} | {wr_at*100:+.2f}%')
    print(f'  H+T avg:     {p_ht} | {wr_ht*100:+.2f}%')
    print(f'  3model avg:  {p_3avg} | {wr_3avg*100:+.2f}%')
    print(f'  3model cons: {p_cons3} | {wr_cons3*100:+.2f}%')
    print(f'  3model wtd:  {p_w} | {wr_w*100:+.2f}%')

print('\nDone!')
