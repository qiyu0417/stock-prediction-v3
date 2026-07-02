"""MC=20 eval: Tech39 standalone + 3-model ensemble (Alpha158 + Tech39 + Hybrid)"""
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

# Preprocess full data
df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed['instrument'] = processed['股票代码'].map(sid2idx); processed = processed.dropna(subset=['instrument']).copy()
processed['instrument'] = processed['instrument'].astype(np.int64)
processed = _build_label_and_clean(processed, drop_small_open=True)

# Feature column lists
alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]
tech_f = [f for f in _TECH_39_ONLY if f in fcols_all]
print(f"Features: full={len(fcols_all)} alpha={len(alpha_f)} tech={len(tech_f)}")

import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__; _orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

def load_models(mdir, fd, ns):
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

ns = cfg = None
print('Loading models...')
# Hybrid
with open('model/stock_emb_8_hybrid/ensemble_config.json', 'r') as f: cfg = json.load(f)
fd_full = cfg['feature_dim']; ns = cfg['num_stocks']
H = load_models('model/stock_emb_8_hybrid', fd_full, ns)
# Alpha158
A = load_models('model/stock_emb_8_alpha158', len(alpha_f), ns)
# Tech39
T = load_models('model/stock_emb_8_tech39', len(tech_f), ns)
print(f'Models: H={len(H)} A={len(A)} T={len(T)}')

# Preprocess: each model has its own winsor+scaler fitted on full 197 features
# Apply winsor+scaler to ALL features first, then select subsets
def preprocess_all(scaler_path, winsor_path):
    p = processed.copy()
    p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan)
    p = p.dropna(subset=fcols_all)
    with open(winsor_path, 'r') as f: wb = json.load(f)
    for col, (lo, hi) in wb.items():
        if col in p.columns: p[col] = p[col].clip(lo, hi)
    sc = joblib.load(scaler_path)
    p[fcols_all] = sc.transform(p[fcols_all])
    return p

print('Preprocessing feature sets...')
p_full = preprocess_all('model/stock_emb_8_hybrid/scaler.pkl', 'model/stock_emb_8_hybrid/winsor_bounds.json')
p_alpha = preprocess_all('model/stock_emb_8_alpha158/scaler.pkl', 'model/stock_emb_8_alpha158/winsor_bounds.json')
p_tech = preprocess_all('model/stock_emb_8_tech39/scaler.pkl', 'model/stock_emb_8_tech39/winsor_bounds.json')

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

def build_seq(hist, sids, fcols, p_df, n_feat):
    seq = np.zeros((1, len(sids), SEQ, n_feat), dtype=np.float32)
    valid = np.zeros(len(sids), dtype=bool)
    for i, sid in enumerate(sids):
        sd = p_df[(p_df['股票代码'] == sid) & (p_df['日期'] <= pd.to_datetime(hist))].sort_values('日期')
        if len(sd) >= SEQ:
            seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32)
            valid[i] = True
    return seq, valid

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

print('\n=== Tech39 + 3-Model Smart Ensemble (MC=20) ===')
results = {}

for pd_str in ['2026-06-01', '2026-06-08']:
    hist = processed[processed['日期'] <= pd_str]
    sids = sorted(hist['股票代码'].unique())
    n_stocks = len(sids)

    # Build seq tensors for each model
    seq_full, vf = build_seq(pd_str, sids, fcols_all, p_full, fd_full)
    seq_alpha, va = build_seq(pd_str, sids, alpha_f, p_alpha, len(alpha_f))
    seq_tech, vt = build_seq(pd_str, sids, tech_f, p_tech, len(tech_f))

    st_full = torch.FloatTensor(seq_full).to(device)
    st_alpha = torch.FloatTensor(seq_alpha).to(device)
    st_tech = torch.FloatTensor(seq_tech).to(device)

    # MC scores
    sc_h = mc_scores(H, st_full)
    sc_a = mc_scores(A, st_alpha)
    sc_t = mc_scores(T, st_tech)

    # Standalone
    wr_h, p_h = compute_week(pd_str, sc_h, vf, sids, 'Hybrid')
    wr_a, p_a = compute_week(pd_str, sc_a, va, sids, 'Alpha')
    wr_t, p_t = compute_week(pd_str, sc_t, vt, sids, 'Tech39')

    print(f'\n{pd_str}:')
    print(f'  Hybrid:     {p_h} | {wr_h*100:+.2f}%')
    print(f'  Alpha158:   {p_a} | {wr_a*100:+.2f}%')
    print(f'  Tech39:     {p_t} | {wr_t*100:+.2f}%')

    # 2-model ensembles
    sc_ah = (sc_a + sc_h) / 2.0
    wr_ah, p_ah = compute_week(pd_str, sc_ah, vf, sids)
    sc_at = (sc_a + sc_t) / 2.0
    wr_at, p_at = compute_week(pd_str, sc_at, vt, sids)
    sc_ht = (sc_h + sc_t) / 2.0
    wr_ht, p_ht = compute_week(pd_str, sc_ht, vf, sids)

    # 3-model simple average
    sc_3avg = (sc_h + sc_a + sc_t) / 3.0
    wr_3avg, p_3avg = compute_week(pd_str, sc_3avg, vf, sids)

    # Consensus filter: keep stocks in top-10 of at least 2 of 3 models
    top_h = set(np.argsort(sc_h)[-10:])
    top_a = set(np.argsort(sc_a)[-10:])
    top_t = set(np.argsort(sc_t)[-10:])
    consensus_3 = []
    for idx in range(n_stocks):
        count = (idx in top_h) + (idx in top_a) + (idx in top_t)
        if count >= 2:
            consensus_3.append(idx)
    sc_cons3 = sc_h.copy()
    if consensus_3:
        sc_cons3[consensus_3] = sc_h[consensus_3] * 1.10
    wr_cons3, p_cons3 = compute_week(pd_str, sc_cons3, vf, sids)

    # Weighted: Alpha (W1 best) + Hybrid (W2 best) + Tech (diversifier)
    sc_weighted = 0.35 * sc_a + 0.45 * sc_h + 0.20 * sc_t
    wr_w, p_w = compute_week(pd_str, sc_weighted, vf, sids)

    print(f'  A+H 2-model:  {p_ah} | {wr_ah*100:+.2f}%')
    print(f'  A+T 2-model:  {p_at} | {wr_at*100:+.2f}%')
    print(f'  H+T 2-model:  {p_ht} | {wr_ht*100:+.2f}%')
    print(f'  3-model avg:  {p_3avg} | {wr_3avg*100:+.2f}%')
    print(f'  3-model cons: {p_cons3} | {wr_cons3*100:+.2f}%')
    print(f'  3-model wtd:  {p_w} | {wr_w*100:+.2f}%')

print('\nDone!')
