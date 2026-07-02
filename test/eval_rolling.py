"""
Rolling-window adaptive weights: for each test week, find optimal w
based on past N training days, apply to current week. MC=20 + 5 seeds.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np
import pandas as pd
import torch
import joblib

MC_TEST = 20
SEQ = 60

# ── Load data ──
print("Loading data...")
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6)
train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')
new_df = pd.read_csv('data/new_week.csv', dtype={'股票代码': str})
new_df['股票代码'] = new_df['股票代码'].astype(str).str.zfill(6)
new_df['日期'] = pd.to_datetime(new_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df, new_df]).drop_duplicates(
    subset=['股票代码', '日期'], keep='last')
full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
all_sids = sorted(full_df['股票代码'].unique())
sid2idx = {s: i for i, s in enumerate(all_sids)}

from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate
from config_stock_emb_8 import FEATURE_NUM

fe = feature_engineer_func_map[FEATURE_NUM]
fcols_all = feature_cloums_map[FEATURE_NUM]
groups = [g.reset_index(drop=True) for _, g in
          full_df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument'] = processed_raw['股票代码'].map(sid2idx)
processed_raw = processed_raw.dropna(subset=['instrument']).copy()
processed_raw['instrument'] = processed_raw['instrument'].astype(np.int64)
processed_raw = _build_label_and_clean(processed_raw, drop_small_open=True)
alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]

# ── Load models ──
import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__
_orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

device = torch.device('cuda')


def load_model(mdir, nf):
    _em.StockTransformerExpert.__init__ = _orig_ti
    _em.ConvStockExpert.__init__ = _orig_ci
    with open(os.path.join(mdir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    with open('model/stock_emb_8_hybrid/ensemble_config.json') as f2:
        ns = json.load(f2)['num_stocks']
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        e = dict(ec)
        e['stock_embed_dim'] = emb
        m = StockTransformerExpert(nf, e, ns) if ec['type'] == 'transformer' \
            else ConvStockExpert(nf, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        models.append(m)
    return models


def preprocess(mdir):
    with open(os.path.join(mdir, 'winsor_bounds.json')) as f:
        wb = json.load(f)
    sc = joblib.load(os.path.join(mdir, 'scaler.pkl'))
    p = processed_raw.copy()
    p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
    for col, (lo, hi) in wb.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    p[fcols_all] = sc.transform(p[fcols_all])
    return p


print("Loading models...")
H = load_model('model/stock_emb_8_hybrid', len(fcols_all))
A = load_model('model/stock_emb_8_alpha158', len(alpha_f))
p_h = preprocess('model/stock_emb_8_hybrid')
p_a = preprocess('model/stock_emb_8_alpha158')
print(f"  H={len(H)} experts, A={len(A)} experts")


def build_seq(p, ref_date, fcols, nf):
    hist = p[p['日期'] <= ref_date]
    sids = sorted(hist['股票代码'].unique())
    seq = np.zeros((1, len(sids), SEQ, nf), dtype=np.float32)
    valid = np.zeros(len(sids), dtype=bool)
    for i, sid in enumerate(sids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ:
            seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32)
            valid[i] = True
    return seq, sids, valid


def mc_infer(models, seq_t, n_samples):
    all_s = []
    for _ in range(n_samples):
        ps = []
        for m in models:
            with torch.no_grad():
                out = m(seq_t)
                if isinstance(out, tuple):
                    out = out[0]
                ps.append(out[0].cpu().numpy())
        all_s.append(np.mean(ps, axis=0))
    return np.mean(all_s, axis=0)


# ── Load training inference cache ──
cache = joblib.load('data/blend_data_v2.pkl')
train_data = cache['train_data']
print(f"Loaded {len(train_data)} training days from cache")

# Sort by date
train_data.sort(key=lambda x: x['date'])


# ═══════════════════════════════════════════════════════════
# Rolling window: find best w based on past N training days
# ═══════════════════════════════════════════════════════════
def find_best_w_rolling(ref_date_str, lookback_days=20):
    """Find optimal w using past N training days before ref_date_str."""
    ref_date = pd.Timestamp(ref_date_str)
    candidates = [td for td in train_data
                  if pd.Timestamp(td['date']) < ref_date]
    if len(candidates) > lookback_days:
        candidates = candidates[-lookback_days:]

    if len(candidates) < 5:
        return 0.5  # default

    best_w, best_ret = 0.5, -float('inf')
    for w in np.arange(0.0, 1.05, 0.05):
        w = round(w, 2)
        daily_rets = []
        for td in candidates:
            common = td['common']
            combined = {}
            for sid in common:
                combined[sid] = (w * td['h_scores'].get(sid, 0) +
                                 (1 - w) * td['a_scores'].get(sid, 0))

            filt = td['filt']
            if len(filt) < 5:
                continue
            bnc = td['bnc']
            qual = td['qual']

            final = {}
            for sid in filt:
                s = combined.get(sid, -999)
                if sid not in bnc:
                    s *= 0.92
                s += (qual.get(sid, 0.5) - 0.5) * 0.05
                final[sid] = s

            top5 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
            sids_top = [s for s, _ in top5]
            _, weights = equal_weight_allocate(sids_top)
            ret = 0.0
            for sid, wgt in zip(sids_top, weights):
                r1 = full_df[(full_df['股票代码'] == sid) &
                             (full_df['日期'] == td['t1_date'])]
                r5 = full_df[(full_df['股票代码'] == sid) &
                             (full_df['日期'] == td['t5_date'])]
                if len(r1) > 0 and len(r5) > 0:
                    sr = (float(r5.iloc[0]['开盘']) - float(r1.iloc[0]['开盘'])) / float(
                        r1.iloc[0]['开盘'])
                    ret += sr * wgt
            daily_rets.append(ret)

        if daily_rets:
            mean_ret = np.mean(daily_rets)
            if mean_ret > best_ret:
                best_ret = mean_ret
                best_w = w

    return best_w


def compute_top5_return(top5_sids, t1_date, t5_date):
    sids, _ = equal_weight_allocate(top5_sids)
    ret = 0.0
    for sid, wgt in zip(sids, [0.2] * 5):
        r1 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t1_date)]
        r5 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t5_date)]
        if len(r1) > 0 and len(r5) > 0:
            sr = (float(r5.iloc[0]['开盘']) - float(r1.iloc[0]['开盘'])) / float(r1.iloc[0]['开盘'])
            ret += sr * wgt
    return ret


# ═══════════════════════════════════════════════════════════
# Evaluation on June test set
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MC=20 5-seed evaluation with rolling-window weights")
print(f"{'='*60}")

weeks = [
    ('W1', pd.to_datetime('2026-05-29'), pd.to_datetime('2026-06-01'),
     pd.to_datetime('2026-06-05'), 5),
    ('W2', pd.to_datetime('2026-06-05'), pd.to_datetime('2026-06-08'),
     pd.to_datetime('2026-06-12'), 5),
    ('W3', pd.to_datetime('2026-06-12'), pd.to_datetime('2026-06-15'),
     pd.to_datetime('2026-06-18'), 4),
]

# Find rolling weights for each week (deterministic, based on past data)
rolling_weights = {}
for wname, pd_str, _, _, _ in weeks:
    w_roll = find_best_w_rolling(str(pd_str.date()))
    rolling_weights[wname] = w_roll
    print(f"  {wname} rolling w (past 20d): {w_roll:.2f}")

# Evaluate with different window sizes
window_sizes = [5, 10, 20, 30, 50]

all_results = {}
for n_days in window_sizes:
    # Find rolling weights for each week
    ws = {}
    for wname, pd_str, _, _, _ in weeks:
        ws[wname] = find_best_w_rolling(str(pd_str.date()), n_days)
    print(f"\n  n={n_days}: w_W1={ws['W1']:.2f} w_W2={ws['W2']:.2f} w_W3={ws['W3']:.2f}")

seeds = [42, 123, 456, 789, 1024]
all_results = []

for seed in seeds:
    set_seed(seed)
    print(f"\n--- Seed={seed} ---")

    for wname, pd_str, t1_date, t5_date, ndays in weeks:
        ref_date = pd_str
        ref_str = str(pd_str.date())

        seq_h, sids_h, valid_h = build_seq(p_h, ref_date, fcols_all, len(fcols_all))
        seq_a, sids_a, valid_a = build_seq(p_a, ref_date, alpha_f, len(alpha_f))

        raw_h = mc_infer(H, torch.FloatTensor(seq_h).to(device), MC_TEST)
        raw_a = mc_infer(A, torch.FloatTensor(seq_a).to(device), MC_TEST)

        raw_hist = processed_raw[processed_raw['日期'] <= ref_date]
        h_map = {s: float(raw_h[i]) for i, s in enumerate(sids_h) if valid_h[i]}
        a_map = {s: float(raw_a[i]) for i, s in enumerate(sids_a) if valid_a[i]}
        common = sorted(set(h_map.keys()) & set(a_map.keys()))

        for strat_name, w in [
            ('Rolling(n=20)', find_best_w_rolling(ref_str, 20)),
            ('Rolling(n=10)', find_best_w_rolling(ref_str, 10)),
            ('Rolling(n=5)', find_best_w_rolling(ref_str, 5)),
            ('Fixed(w=0.45)', 0.45),
            ('Fixed(w=0.5)', 0.5),
            ('Hybrid only', 1.0),
            ('Alpha158 only', 0.0),
        ]:
            combined = {sid: w * h_map.get(sid, 0) + (1 - w) * a_map.get(sid, 0)
                         for sid in common}

            filt = volatility_filter(raw_hist, list(common), ref_str, top_pct=0.95)
            if len(filt) < 5:
                continue
            bnc = bounce_confirm(raw_hist, filt, ref_str, threshold=0.008)
            qual = compute_quality_score(raw_hist, filt, ref_str)

            final = {}
            for sid in filt:
                s = combined.get(sid, -999)
                if sid not in bnc:
                    s *= 0.92
                s += (qual.get(sid, 0.5) - 0.5) * 0.05
                final[sid] = s

            top5 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
            ret = compute_top5_return([s for s, _ in top5], t1_date, t5_date)

            all_results.append({
                'Seed': seed, 'Week': wname, 'Strategy': strat_name,
                'Return': ret, 'w': w,
                'Top5': ','.join([s for s, _ in top5]),
            })

# ── Summary ──
df = pd.DataFrame(all_results)
print(f"\n{'='*70}")
print("RESULTS: MC=20 5-seed mean returns")
print(f"{'='*70}")

strategies = df['Strategy'].unique()
for strat in sorted(strategies):
    sub = df[df['Strategy'] == strat]
    mean_ret = sub['Return'].mean() * 100
    std_ret = sub['Return'].std() * 100
    print(f"  {strat:20s}: {mean_ret:+.2f}% ±{std_ret:.2f}%")

print(f"\n{'='*70}")
print("WEEKLY BREAKDOWN")
print(f"{'='*70}")
for week_name, _, _, _, _ in weeks:
    print(f"\n  {week_name}:")
    sub_w = df[df['Week'] == week_name]
    for strat in sorted(strategies):
        sub = sub_w[sub_w['Strategy'] == strat]
        if len(sub) > 0:
            ws = sub['w'].values
            print(f"    {strat:20s}: {sub['Return'].mean()*100:+.2f}% "
                  f"±{sub['Return'].std()*100:.2f}% (w={ws.mean():.2f})")

print(f"\nDone!")
