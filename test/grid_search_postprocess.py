"""Grid search post-processing — FIXED: use processed (feature-engineered) data, not raw CSV"""
import os, sys, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda'); set_seed(42)

# --- data ---
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str}); train_df['股票代码']=train_df['股票代码'].astype(str).str.zfill(6); train_df['日期']=pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str}); test_df['股票代码']=test_df['股票代码'].astype(str).str.zfill(6); test_df['日期']=pd.to_datetime(test_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=['股票代码', '日期'], keep='last')
test_dates = sorted(test_df['日期'].unique())
all_sids = sorted(full_df['股票代码'].unique()); sid2idx = {s:i for i,s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]; fcols_all = feature_cloums_map[FEATURE_NUM]

# Build processed data (raw + scaled)
df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument'] = processed_raw['股票代码'].map(sid2idx); processed_raw = processed_raw.dropna(subset=['instrument']).copy()
processed_raw['instrument'] = processed_raw['instrument'].astype(np.int64)
processed_raw = _build_label_and_clean(processed_raw, drop_small_open=True)
# Save raw (un-winsorized, un-scaled) for post-processing
processed_raw = processed_raw.copy()

# Scaled version for model input
base_dir = 'model/stock_emb_8_hybrid'
with open(os.path.join(base_dir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
fd_full = cfg['feature_dim']; ns = cfg['num_stocks']
scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
with open(os.path.join(base_dir, 'winsor_bounds.json'), 'r') as f: wb = json.load(f)

processed = processed_raw.copy()
processed[fcols_all] = processed[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
for col, (lo, hi) in wb.items():
    if col in processed.columns: processed[col] = processed[col].clip(lo, hi)
processed[fcols_all] = scaler.transform(processed[fcols_all])

# --- load model & compute MC=20 scores once ---
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

CACHE = 'logs/mc20_scores_cache.npz'
if os.path.exists(CACHE):
    print('Loading cached MC=20 scores...')
    cached = np.load(CACHE, allow_pickle=True)
    week_data = {}
    for pd_str in ['2026-06-01', '2026-06-08']:
        week_data[pd_str] = {
            'scores': cached[f'{pd_str}_scores'],
            'valid': cached[f'{pd_str}_valid'],
            'sids': cached[f'{pd_str}_sids']
        }
else:
    print('Loading Hybrid model...')
    H = load_models(base_dir, fd_full)
    print(f'  {len(H)} experts loaded')
    print('Computing MC=20 scores...')
    week_data = {}
    for pd_str in ['2026-06-01', '2026-06-08']:
        print(f'  {pd_str}...')
        hist = processed[processed['日期'] <= pd_str]
        sids = sorted(hist['股票代码'].unique()); n_stocks = len(sids)
        seq_full = np.zeros((1, n_stocks, SEQ, fd_full), dtype=np.float32)
        valid = np.zeros(n_stocks, dtype=bool)
        for i, sid in enumerate(sids):
            sd = hist[hist['股票代码'] == sid].sort_values('日期')
            if len(sd) >= SEQ:
                seq_full[0, i] = sd[fcols_all].values[-SEQ:].astype(np.float32)
                valid[i] = True
        seq_t = torch.FloatTensor(seq_full).to(device)
        all_s = []
        for _ in range(MC):
            ps = []
            for m in H:
                with torch.no_grad():
                    p = m(seq_t)
                    if isinstance(p, tuple): p = p[0]
                    ps.append(p[0].cpu().numpy())
            all_s.append(np.mean(ps, axis=0))
        sc = np.mean(all_s, axis=0)
        week_data[pd_str] = {'scores': sc, 'valid': valid, 'sids': sids}
    np.savez(CACHE, **{f'{k}_{kk}': vv for k, v in week_data.items() for kk, vv in v.items()})

# --- precompute post-processing metrics from RAW (un-scaled) data ---
print('Precomputing stock metrics from processed_raw...')

def precompute_metrics_fast(data, sids_set):
    d = data[data['股票代码'].isin(sids_set)].sort_values(['股票代码', '日期'])
    volatilities = {}
    bounce_2d_ret = {}
    quality = {}
    for sid, grp in d.groupby('股票代码'):
        if len(grp) < 3:
            volatilities[sid] = 0.03; bounce_2d_ret[sid] = -999.0; quality[sid] = 0.5; continue
        if 'volatility_20' in grp.columns:
            vv = grp['volatility_20'].dropna()
            volatilities[sid] = float(vv.iloc[-1]) if len(vv) > 0 else 0.03
        else:
            volatilities[sid] = 0.03
        if 'return_1' in grp.columns:
            bounce_2d_ret[sid] = float(grp['return_1'].tail(2).sum())
        else:
            bounce_2d_ret[sid] = -999.0
        rets = grp['return_1'].tail(20).dropna() if 'return_1' in grp.columns else pd.Series(dtype=float)
        if len(rets) < 10:
            quality[sid] = 0.5
        else:
            rv = rets.values
            up_days = (rv > 0).sum()
            consistency = up_days / len(rv)
            mean_ret = rv.mean(); std_ret = rv.std()
            stability = mean_ret / max(std_ret, 0.001)
            cs = 0; max_cs = 0
            for r in rv:
                if r > 0: cs += 1; max_cs = max(max_cs, cs)
                else: cs = 0
            smoothness = max_cs / max(len(rv), 1)
            quality[sid] = consistency * 0.4 + max(0, min(1, stability * 0.5)) * 0.35 + smoothness * 0.25
    return volatilities, bounce_2d_ret, quality

metrics = {}
for pd_str in ['2026-06-01', '2026-06-08']:
    data_cut = processed_raw[processed_raw['日期'] <= pd_str]
    metrics[pd_str] = precompute_metrics_fast(data_cut, set(week_data[pd_str]['sids']))
    v = metrics[pd_str][0]
    print(f'  {pd_str}: {len(v)} stocks, vol range [{min(v.values()):.4f}, {max(v.values()):.4f}]')

# --- vectorized eval using REAL processed data ---
def eval_week_fast(pd_str, sc, valid, sids, vol_pct, bounce_thresh, bounce_penalty, quality_coef):
    volatilities, bounce_2d_ret, quality = metrics[pd_str]
    n = len(sids)
    final = np.full(n, -np.inf, dtype=np.float32)

    if vol_pct < 1.0:
        vol_vals = list(volatilities.values())
        vol_threshold = np.percentile(vol_vals, vol_pct * 100)
    else:
        vol_threshold = float('inf')

    for i, sid in enumerate(sids):
        if not valid[i]:
            continue
        # volatility filter
        if volatilities.get(sid, 0.03) > vol_threshold:
            continue
        s = float(sc[i])
        # bounce confirm
        ret2 = bounce_2d_ret.get(sid, -999.0)
        if ret2 <= bounce_thresh:
            s *= bounce_penalty
        # quality adjustment
        q = quality.get(sid, 0.5)
        s += (q - 0.5) * quality_coef
        final[i] = s

    top_idx = np.argsort(final)[-5:]
    top_sids = [sids[i] for i in top_idx if final[i] > -1e9]
    if len(top_sids) < 5:
        remaining = [i for i in range(n) if final[i] > -1e9 and i not in top_idx]
        for i in remaining[:5-len(top_sids)]:
            top_sids.append(sids[i])
    top_sids = top_sids[:5]
    if len(top_sids) < 5:
        return -1.0, top_sids

    _, weights = equal_weight_allocate(top_sids)
    t1 = next(d for d in test_dates if d >= pd.to_datetime(pd_str))
    t5r = [d for d in test_dates if d >= pd.to_datetime(pd_str)]; t5 = t5r[min(4, len(t5r)-1)]
    ret = 0.0
    for sid, w in zip(top_sids, weights):
        r1 = test_df[(test_df['股票代码']==sid)&(test_df['日期']==t1)]
        r5 = test_df[(test_df['股票代码']==sid)&(test_df['日期']==t5)]
        if len(r1)>0 and len(r5)>0:
            ret += (float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])*w
    return ret, top_sids

# --- grid search ---
vol_pcts = [0.85, 0.88, 0.90, 0.92, 0.93, 0.95, 0.97, 1.0]
bounce_thresholds = [-0.10, -0.05, -0.02, 0.0, 0.005, 0.008, 0.012, 0.020, 0.030]
bounce_penalties = [0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.0]
quality_coefs = [0.0, 0.03, 0.05, 0.10, 0.15]

# Baseline (with REAL processed data)
b_w1, b_p1 = eval_week_fast('2026-06-01', week_data['2026-06-01']['scores'],
    week_data['2026-06-01']['valid'], week_data['2026-06-01']['sids'], 0.95, 0.008, 0.92, 0.05)
b_w2, b_p2 = eval_week_fast('2026-06-08', week_data['2026-06-08']['scores'],
    week_data['2026-06-08']['valid'], week_data['2026-06-08']['sids'], 0.95, 0.008, 0.92, 0.05)

# Also compute NO-FILTER baseline (raw model scores only)
b0_w1, b0_p1 = eval_week_fast('2026-06-01', week_data['2026-06-01']['scores'],
    week_data['2026-06-01']['valid'], week_data['2026-06-01']['sids'], 1.0, -999, 1.0, 0.0)
b0_w2, b0_p2 = eval_week_fast('2026-06-08', week_data['2026-06-08']['scores'],
    week_data['2026-06-08']['valid'], week_data['2026-06-08']['sids'], 1.0, -999, 1.0, 0.0)

print(f'\n=== Baselines ===')
print(f'Old default (vol=0.95, bounce=0.008, penalty=0.92, qual=0.05):')
print(f'  W1: {b_p1} | {b_w1*100:+.2f}%  W2: {b_p2} | {b_w2*100:+.2f}%  Avg: {(b_w1+b_w2)/2*100:+.2f}%')
print(f'No filter (raw model scores only):')
print(f'  W1: {b0_p1} | {b0_w1*100:+.2f}%  W2: {b0_p2} | {b0_w2*100:+.2f}%  Avg: {(b0_w1+b0_w2)/2*100:+.2f}%')

total = len(vol_pcts) * len(bounce_thresholds) * len(bounce_penalties) * len(quality_coefs)
print(f'\nGrid: {total} combinations...')

results = []
count = 0
for vp, bt, bp, qc in itertools.product(vol_pcts, bounce_thresholds, bounce_penalties, quality_coefs):
    wr1, _ = eval_week_fast('2026-06-01', week_data['2026-06-01']['scores'],
        week_data['2026-06-01']['valid'], week_data['2026-06-01']['sids'], vp, bt, bp, qc)
    wr2, _ = eval_week_fast('2026-06-08', week_data['2026-06-08']['scores'],
        week_data['2026-06-08']['valid'], week_data['2026-06-08']['sids'], vp, bt, bp, qc)
    avg = (wr1 + wr2) / 2
    results.append((avg, wr1, wr2, vp, bt, bp, qc))
    count += 1
    if count % 500 == 0:
        print(f'  {count}/{total}...')

results.sort(key=lambda x: x[0], reverse=True)

print(f'\n=== Top 20 Parameter Sets ===')
print(f'{"Rank":<5} {"Avg":>8} {"W1":>8} {"W2":>8} {"vol_pct":>8} {"bounce_th":>10} {"penalty":>8} {"qual_coef":>10}')
print('-' * 85)
for i, (avg, w1, w2, vp, bt, bp, qc) in enumerate(results[:20]):
    print(f'{i+1:<5} {avg*100:+7.2f}% {w1*100:+7.2f}% {w2*100:+7.2f}% {vp:>8.2f} {bt:>10.3f} {bp:>8.2f} {qc:>10.2f}')

# Baseline rank
for i, (avg, w1, w2, vp, bt, bp, qc) in enumerate(results):
    if abs(vp - 0.95) < 0.001 and abs(bt - 0.008) < 0.001 and abs(bp - 0.92) < 0.001 and abs(qc - 0.05) < 0.001:
        print(f'\nOld-default rank: {i+1}/{total} (avg={(w1+w2)/2*100:+.2f}%)')
        break

# Sensitivity
print(f'\n=== Sensitivity Analysis ===')
for param_name, param_values, idx in [
    ('vol_pct', vol_pcts, 3), ('bounce_threshold', bounce_thresholds, 4),
    ('bounce_penalty', bounce_penalties, 5), ('quality_coef', quality_coefs, 6)]:
    print(f'\n{param_name}:')
    for pv in param_values:
        subset = [r for r in results if abs(r[idx] - pv) < 0.001]
        if subset:
            avg_mean = np.mean([r[0] for r in subset])
            avg_best = max([r[0] for r in subset])
            print(f'  {pv:>8.2f}: mean={avg_mean*100:+5.2f}%  best={avg_best*100:+5.2f}%')

# Show baseline (no-filter) for comparison
print(f'\nNo-filter comparison: avg={(b0_w1+b0_w2)/2*100:+.2f}%')

print('\nDone!')
