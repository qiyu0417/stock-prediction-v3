"""
Generate training data for meta-model: for each training day in Jan-May 2026,
grid-search the optimal weight w blending Hybrid + Alpha158 raw scores.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np
import pandas as pd
import torch
import joblib
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate
from market_regime import compute_market_regime

MC = 5
SEQ = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)

VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

print(f"Device: {device} | MC={MC}")

# ── Load and prepare data ──
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
print(f"  Full features: {len(fcols_all)}, Alpha158 features: {len(alpha_f)}")

# ── Load models ──
print("Loading models...")
import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__
_orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert


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
        m.train()  # MC dropout mode
        models.append(m)
    return models


def preprocess(mdir):
    """Preprocess on full feature set (scaler was fit on all 197 features)."""
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


H = load_model('model/stock_emb_8_hybrid', len(fcols_all))
p_h = preprocess('model/stock_emb_8_hybrid')
A = load_model('model/stock_emb_8_alpha158', len(alpha_f))
p_a = preprocess('model/stock_emb_8_alpha158')

print(f"  Hybrid: {len(H)} experts, Alpha158: {len(A)} experts")


# ── Helper: MC inference ──
def mc_infer(models, seq_t, n_samples=MC):
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


# ── Helper: build sequence tensor for a reference date ──
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


# ── Helper: compute actual return for top-5 picks ──
def compute_top5_return(top5, t1_date, t5_date, data_df):
    sids = [s for s, _ in top5]
    _, weights = equal_weight_allocate(sids)
    ret = 0.0
    for sid, w in zip(sids, weights):
        r1 = data_df[(data_df['股票代码'] == sid) & (data_df['日期'] == t1_date)]
        r5 = data_df[(data_df['股票代码'] == sid) & (data_df['日期'] == t5_date)]
        if len(r1) > 0 and len(r5) > 0:
            sr = (float(r5.iloc[0]['开盘']) - float(r1.iloc[0]['开盘'])) / float(r1.iloc[0]['开盘'])
            ret += sr * w
    return ret


# ── Helper: compute all market features at reference date ──
def compute_market_features(ref_date):
    regime = compute_market_regime(processed_raw, fcols_all, all_sids, ref_date)

    daily = processed_raw[processed_raw['日期'] <= ref_date].tail(300 * len(all_sids))
    by_date = daily.groupby('日期')

    if '涨跌幅' in daily.columns:
        market_return_daily = by_date['涨跌幅'].mean()
        recent = market_return_daily.tail(30)
        ret_5d = float(recent.tail(5).sum()) if len(recent) >= 5 else 0.0
        ret_10d = float(recent.tail(10).sum()) if len(recent) >= 10 else 0.0
        ret_20d = float(recent.tail(20).sum()) if len(recent) >= 20 else 0.0
        market_up_ratio = float((recent.tail(5) > 0).mean()) if len(recent) >= 5 else 0.5
        market_return_5d = float(recent.tail(5).mean()) if len(recent) >= 5 else 0.0
        market_vol_5d = float(recent.tail(5).std()) if len(recent) >= 5 else 0.0
    else:
        ret_5d = ret_10d = ret_20d = 0.0
        market_up_ratio = 0.5
        market_return_5d = 0.0
        market_vol_5d = 0.0

    breadth = regime.get('breadth_crash', False)
    ratio_ma20 = 0.5
    if breadth:
        ratio_ma20 = 0.05

    features = {
        'trend_score': regime.get('trend_score', 0.5),
        'breadth_score': regime.get('breadth_score', 0.5),
        'accel_decline_score': regime.get('accel_decline_score', 0.0),
        'volatility_score': regime.get('volatility_score', 0.0),
        'composite': regime.get('composite', 0.5),
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'ret_20d': ret_20d,
        'ratio_ma20': ratio_ma20,
        'vol_20': 0.0,
        'vol_60': 0.0,
        'market_up_ratio_5d': market_up_ratio,
        'market_return_5d': market_return_5d,
        'market_volatility_5d': market_vol_5d,
        'consecutive_downs': 1.0 if regime.get('consecutive_downs', False) else 0.0,
    }
    return features


# ── Main: iterate over training dates ──
all_dates = sorted(full_df['日期'].unique())
train_dates = [d for d in all_dates
               if pd.Timestamp('2026-01-02') <= d <= pd.Timestamp('2026-05-27')]
print(f"Training dates: {len(train_dates)} ({train_dates[0].date()} to {train_dates[-1].date()})")

rows = []
n_valid = 0

for i, ref_date in enumerate(train_dates):
    date_idx = all_dates.index(ref_date)
    if date_idx + 5 >= len(all_dates):
        continue  # need T+5 data

    t1_date = all_dates[date_idx + 1]
    t5_date = all_dates[date_idx + 5]

    # Build sequences for both models
    seq_h, sids_h, valid_h = build_seq(p_h, ref_date, fcols_all, len(fcols_all))
    seq_a, sids_a, valid_a = build_seq(p_a, ref_date, alpha_f, len(alpha_f))

    if len(sids_h) < 10:
        continue

    # MC inference
    seq_ht = torch.FloatTensor(seq_h).to(device)
    raw_h = mc_infer(H, seq_ht)
    seq_at = torch.FloatTensor(seq_a).to(device)
    raw_a = mc_infer(A, seq_at)

    # Build score maps
    h_map = {s: float(raw_h[i]) for i, s in enumerate(sids_h) if valid_h[i]}
    a_map = {s: float(raw_a[i]) for i, s in enumerate(sids_a) if valid_a[i]}

    # Compute market features
    mkt = compute_market_features(ref_date)

    # Grid search optimal w
    best_w, best_ret = 0.5, -float('inf')
    w_returns = {}

    # Common post-processing data
    raw_hist = processed_raw[processed_raw['日期'] <= ref_date]
    ref_str = str(ref_date.date())

    for w in np.arange(0.0, 1.05, 0.1):
        w = round(w, 2)
        common_sids = sorted(set(h_map.keys()) & set(a_map.keys()))
        if len(common_sids) < 10:
            continue

        combined = {sid: w * h_map.get(sid, 0) + (1 - w) * a_map.get(sid, 0)
                     for sid in common_sids}

        filt = volatility_filter(raw_hist, list(common_sids), ref_str, top_pct=VP)
        if len(filt) < 5:
            continue
        bnc = bounce_confirm(raw_hist, filt, ref_str, threshold=BT)
        qual = compute_quality_score(raw_hist, filt, ref_str)

        final = {}
        for sid in filt:
            s = combined.get(sid, -999)
            if sid not in bnc:
                s *= BP
            s += (qual.get(sid, 0.5) - 0.5) * QC
            final[sid] = s

        top5 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
        ret = compute_top5_return(top5, t1_date, t5_date, full_df)
        w_returns[w] = ret

        if ret > best_ret:
            best_ret = ret
            best_w = w

    if best_ret == -float('inf'):
        continue

    row = {
        'date': str(ref_date.date()),
        'w_opt': best_w,
        'best_ret': best_ret,
        **mkt,
    }
    # Also record returns for w=0 and w=1 for baseline context
    row['w0_ret'] = w_returns.get(0.0, float('nan'))
    row['w1_ret'] = w_returns.get(1.0, float('nan'))
    rows.append(row)
    n_valid += 1

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(train_dates)}] {ref_date.date()} "
              f"w*={best_w:.1f} ret={best_ret*100:+.2f}% "
              f"H={w_returns.get(1.0, 0)*100:+.2f}% "
              f"A={w_returns.get(0.0, 0)*100:+.2f}%")

print(f"\nValid samples: {n_valid}/{len(train_dates)}")

# ── EMA smooth optimal w ──
df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
df['w_opt_smoothed'] = df['w_opt'].ewm(span=3, adjust=False).mean()

# Fill any NaN at the beginning with raw value
mask = df['w_opt_smoothed'].isna()
df.loc[mask, 'w_opt_smoothed'] = df.loc[mask, 'w_opt']

# ── Save ──
os.makedirs('data', exist_ok=True)
df.to_csv('data/meta_train.csv', index=False)
print(f"Saved data/meta_train.csv ({len(df)} rows)")

# ── Summary stats ──
print(f"\n=== Summary ===")
print(f"w_opt mean: {df['w_opt'].mean():.3f} ± {df['w_opt'].std():.3f}")
print(f"w_opt_smoothed mean: {df['w_opt_smoothed'].mean():.3f} ± {df['w_opt_smoothed'].std():.3f}")
print(f"w=0 (Alpha158) avg ret: {df['w0_ret'].mean()*100:+.2f}%")
print(f"w=1 (Hybrid) avg ret: {df['w1_ret'].mean()*100:+.2f}%")
print(f"best_w avg ret: {df['best_ret'].mean()*100:+.2f}%")
print(f"\nw distribution:")
for w in np.arange(0.0, 1.05, 0.1):
    w = round(w, 2)
    count = (df['w_opt'] == w).sum()
    if count > 0:
        print(f"  w={w:.1f}: {count} days")
print(f"\nDone!")
