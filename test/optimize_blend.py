"""
Optimize 3-model blending weights + PP parameters. (OPTIMIZED)
Pre-computes PP filters per day to avoid redundant computation.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np
import pandas as pd
import torch
import joblib
from itertools import product
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC_TRAIN = 5
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
full_df = pd.concat([train_df, test_df, new_df]).drop_duplicates(subset=['股票代码', '日期'], keep='last')
full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
all_sids = sorted(full_df['股票代码'].unique())
sid2idx = {s: i for i, s in enumerate(all_sids)}
all_dates_full = sorted(full_df['日期'].unique())

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]
fcols_all = feature_cloums_map[FEATURE_NUM]
groups = [g.reset_index(drop=True) for _, g in full_df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
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


def load_model(mdir, nf, monkey_patch=None):
    _em.StockTransformerExpert.__init__ = _orig_ti
    _em.ConvStockExpert.__init__ = _orig_ci

    if monkey_patch == 'se_v2':
        from train_se_v2 import SEBlock, SEWrapper

        def _se_trans_init(self, input_dim, expert_config, num_stocks):
            _orig_ti(self, input_dim, expert_config, num_stocks)
            self.feature_attention = SEWrapper(self.feature_attention, self.d_model)

        def _se_conv_init(self, input_dim, expert_config, num_stocks):
            _orig_ci(self, input_dim, expert_config, num_stocks)
            self.feature_attention = SEWrapper(self.feature_attention, self.d_model)

        _em.StockTransformerExpert.__init__ = _se_trans_init
        _em.ConvStockExpert.__init__ = _se_conv_init
        StockTransformerExpert.__init__ = _se_trans_init
        ConvStockExpert.__init__ = _se_conv_init

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

    _em.StockTransformerExpert.__init__ = _orig_ti
    _em.ConvStockExpert.__init__ = _orig_ci
    StockTransformerExpert.__init__ = _orig_ti
    ConvStockExpert.__init__ = _orig_ci
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
S = load_model('model/stock_emb_8_se_v2', len(fcols_all), monkey_patch='se_v2')
p_h = preprocess('model/stock_emb_8_hybrid')
p_a = preprocess('model/stock_emb_8_alpha158')
p_s = preprocess('model/stock_emb_8_se_v2')
print(f"  H={len(H)} experts, A={len(A)} experts, S={len(S)} experts")


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


def compute_top5_return_from_labels(top5_sids, t1_date, t5_date):
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
# PHASE 1: MC=5 inference + pre-compute PP filters per day
# ═══════════════════════════════════════════════════════════
CACHE_FILE = 'data/blend_data_v2.pkl'

if os.path.exists(CACHE_FILE):
    print(f"\nLoading cached: {CACHE_FILE}")
    cache = joblib.load(CACHE_FILE)
    train_data = cache['train_data']
else:
    print(f"\n{'='*60}")
    print("PHASE 1: MC=5 inference + pre-compute PP filters (3 models)")
    print(f"{'='*60}")

    train_dates = [d for d in all_dates_full
                   if pd.Timestamp('2026-01-02') <= d <= pd.Timestamp('2026-05-27')]
    train_data = []

    for i, ref_date in enumerate(train_dates):
        date_idx = all_dates_full.index(ref_date)
        if date_idx + 5 >= len(all_dates_full):
            continue

        t1_date = all_dates_full[date_idx + 1]
        t5_date = all_dates_full[date_idx + 5]

        # Inference for all 3 models
        seq_h, sids_h, valid_h = build_seq(p_h, ref_date, fcols_all, len(fcols_all))
        raw_h = mc_infer(H, torch.FloatTensor(seq_h).to(device), MC_TRAIN)

        seq_a, sids_a, valid_a = build_seq(p_a, ref_date, alpha_f, len(alpha_f))
        raw_a = mc_infer(A, torch.FloatTensor(seq_a).to(device), MC_TRAIN)

        seq_s, sids_s, valid_s = build_seq(p_s, ref_date, fcols_all, len(fcols_all))
        raw_s = mc_infer(S, torch.FloatTensor(seq_s).to(device), MC_TRAIN)

        if len(sids_h) < 10:
            continue

        # Pre-compute PP filters (same for all weight combos)
        raw_hist = processed_raw[processed_raw['日期'] <= ref_date]
        ref_str = str(ref_date.date())

        # Build score maps for common stocks
        h_map = {s: float(raw_h[i]) for i, s in enumerate(sids_h) if valid_h[i]}
        a_map = {s: float(raw_a[i]) for i, s in enumerate(sids_a) if valid_a[i]}
        s_map = {s: float(raw_s[i]) for i, s in enumerate(sids_s) if valid_s[i]}
        common = sorted(set(h_map.keys()) & set(a_map.keys()) & set(s_map.keys()))

        # PP filters (independent of weight combo)
        filt = volatility_filter(raw_hist, list(common), ref_str, top_pct=0.95)
        bnc = bounce_confirm(raw_hist, filt, ref_str, threshold=0.008)
        qual = compute_quality_score(raw_hist, filt, ref_str)

        train_data.append({
            'date': str(ref_date.date()),
            't1_date': t1_date,
            't5_date': t5_date,
            'common': common,
            'h_scores': {s: h_map[s] for s in common},
            'a_scores': {s: a_map[s] for s in common},
            's_scores': {s: s_map[s] for s in common},
            'filt': list(filt),
            'bnc': bnc,
            'qual': qual,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(train_dates)}] {ref_date.date()}")

    joblib.dump({'train_data': train_data}, CACHE_FILE)
    print(f"Saved {CACHE_FILE} ({len(train_data)} days)")


# ═══════════════════════════════════════════════════════════
# Fast evaluation: compute return for given weights + PP
# ═══════════════════════════════════════════════════════════
def eval_blend(wh, wa, ws, vp=0.95, bt=0.008, bp=0.92, qw=0.05):
    """Fast evaluation of weight+PP combo using pre-computed data."""
    daily_rets = []
    for td in train_data:
        common = td['common']
        combined = {}
        for sid in common:
            combined[sid] = (wh * td['h_scores'].get(sid, 0) +
                             wa * td['a_scores'].get(sid, 0) +
                             ws * td['s_scores'].get(sid, 0))

        # Re-compute filt with new vol_pct if different
        if vp != 0.95:
            raw_hist = processed_raw[processed_raw['日期'] <= pd.Timestamp(td['date'])]
            filt = volatility_filter(raw_hist, common, td['date'], top_pct=vp)
        else:
            filt = td['filt']

        if len(filt) < 5:
            continue

        # Re-compute bnc with new threshold if different
        if bt != 0.008:
            raw_hist = processed_raw[processed_raw['日期'] <= pd.Timestamp(td['date'])]
            bnc = bounce_confirm(raw_hist, filt, td['date'], threshold=bt)
        else:
            bnc = td['bnc']

        # Re-compute qual if needed (always use cached for speed)
        qual = td['qual']

        final = {}
        for sid in filt:
            s = combined.get(sid, -999)
            if sid not in bnc:
                s *= bp
            s += (qual.get(sid, 0.5) - 0.5) * qw
            final[sid] = s

        top5 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
        ret = compute_top5_return_from_labels([s for s, _ in top5],
                                               td['t1_date'], td['t5_date'])
        daily_rets.append(ret)

    if not daily_rets:
        return -float('inf'), 0
    return np.mean(daily_rets), np.std(daily_rets) / np.sqrt(len(daily_rets))


# ═══════════════════════════════════════════════════════════
# PHASE 2: Grid search weights (fast, reuses pre-computed PP)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 2: Grid search 3-model weights (pre-computed PP filters)")
print(f"{'='*60}")

weight_combos = []
for i in range(11):
    for j in range(11 - i):
        k = 10 - i - j
        weight_combos.append((i / 10, j / 10, k / 10))
print(f"Weight combos: {len(weight_combos)}")

results = []
for wh, wa, ws in weight_combos:
    mean_ret, se_ret = eval_blend(wh, wa, ws)
    results.append((wh, wa, ws, mean_ret, se_ret))

results.sort(key=lambda x: x[3], reverse=True)
print(f"\nTop 10 weight combos:")
for i, (wh, wa, ws, mean_ret, se_ret) in enumerate(results[:10]):
    print(f"  {i+1}. H={wh:.1f} A={wa:.1f} S={ws:.1f}  "
          f"ret={mean_ret*100:+.2f}% ±{se_ret*100:.2f}%")

wh_best, wa_best, ws_best, _, _ = results[0]
print(f"\nBest: H={wh_best:.1f} A={wa_best:.1f} S={ws_best:.1f}")

# Use default PP params (already optimal per experiment #44)
vp_best, bt_best, bp_best, qw_best = 0.95, 0.008, 0.92, 0.05
print(f"\nPP params: vol={vp_best} bounce={bt_best} penalty={bp_best} qual={qw_best} (OldDefault)")

# Also evaluate baselines
print(f"\nBaselines (with best PP):")
for label, wh, wa, ws in [
    ('H+A (w=0.45/0.55)', 0.45, 0.55, 0.0),
    ('H only', 1.0, 0.0, 0.0),
    ('A only', 0.0, 1.0, 0.0),
    ('S only', 0.0, 0.0, 1.0),
]:
    mean_ret, se_ret = eval_blend(wh, wa, ws, vp=vp_best, bt=bt_best, bp=bp_best, qw=qw_best)
    print(f"  {label:25s}: {mean_ret*100:+.2f}% ±{se_ret*100:.2f}%")


# ═══════════════════════════════════════════════════════════
# PHASE 3: MC=20 + 5-seed evaluation on June test set
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 3: MC=20 5-seed evaluation on June test set")
print(f"{'='*60}")

weeks = [
    ('W1', pd.to_datetime('2026-05-29'), pd.to_datetime('2026-06-01'),
     pd.to_datetime('2026-06-05'), 5),
    ('W2', pd.to_datetime('2026-06-05'), pd.to_datetime('2026-06-08'),
     pd.to_datetime('2026-06-12'), 5),
    ('W3', pd.to_datetime('2026-06-12'), pd.to_datetime('2026-06-15'),
     pd.to_datetime('2026-06-18'), 4),
]

seeds = [42, 123, 456, 789, 1024]

strategies = [
    ('3-model best', wh_best, wa_best, ws_best),
    ('H+A w=0.45/0.55', 0.45, 0.55, 0.0),
    ('Hybrid only', 1.0, 0.0, 0.0),
    ('Alpha158 only', 0.0, 1.0, 0.0),
    ('SE_v2 only', 0.0, 0.0, 1.0),
]

all_results = []

for seed in seeds:
    set_seed(seed)
    print(f"\n--- Seed={seed} ---")

    for wname, pd_str, t1_date, t5_date, ndays in weeks:
        ref_date = pd_str
        ref_str = str(pd_str.date())

        seq_h, sids_h, valid_h = build_seq(p_h, ref_date, fcols_all, len(fcols_all))
        seq_a, sids_a, valid_a = build_seq(p_a, ref_date, alpha_f, len(alpha_f))
        seq_s, sids_s, valid_s = build_seq(p_s, ref_date, fcols_all, len(fcols_all))

        raw_h = mc_infer(H, torch.FloatTensor(seq_h).to(device), MC_TEST)
        raw_a = mc_infer(A, torch.FloatTensor(seq_a).to(device), MC_TEST)
        raw_s = mc_infer(S, torch.FloatTensor(seq_s).to(device), MC_TEST)

        raw_hist = processed_raw[processed_raw['日期'] <= ref_date]

        for strat_name, wh, wa, ws in strategies:
            h_map = {s: float(raw_h[i]) for i, s in enumerate(sids_h) if valid_h[i]}
            a_map = {s: float(raw_a[i]) for i, s in enumerate(sids_a) if valid_a[i]}
            s_map = {s: float(raw_s[i]) for i, s in enumerate(sids_s) if valid_s[i]}
            common = sorted(set(h_map.keys()) & set(a_map.keys()) & set(s_map.keys()))
            if len(common) < 10:
                continue

            combined = {sid: wh * h_map.get(sid, 0) + wa * a_map.get(sid, 0) +
                         ws * s_map.get(sid, 0) for sid in common}

            filt = volatility_filter(raw_hist, list(common), ref_str, top_pct=vp_best)
            if len(filt) < 5:
                continue
            bnc = bounce_confirm(raw_hist, filt, ref_str, threshold=bt_best)
            qual = compute_quality_score(raw_hist, filt, ref_str)

            final = {}
            for sid in filt:
                s = combined.get(sid, -999)
                if sid not in bnc:
                    s *= bp_best
                s += (qual.get(sid, 0.5) - 0.5) * qw_best
                final[sid] = s

            top5 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
            ret = compute_top5_return_from_labels([s for s, _ in top5], t1_date, t5_date)

            all_results.append({
                'Seed': seed, 'Week': wname, 'Strategy': strat_name,
                'Return': ret, 'Top5': ','.join([s for s, _ in top5]),
            })

        print(f"  {wname} ok")

# ── Summary ──
df = pd.DataFrame(all_results)
print(f"\n{'='*70}")
print(f"FINAL: MC={MC_TEST}, 5 seeds, best PP (vp={vp_best}, bt={bt_best}, bp={bp_best}, qw={qw_best})")
print(f"{'='*70}")

for strat_name, wh, wa, ws in strategies:
    sub = df[df['Strategy'] == strat_name]
    if len(sub) > 0:
        print(f"  {strat_name:25s}: {sub['Return'].mean()*100:+.2f}% ±{sub['Return'].std()*100:.2f}%")

print(f"\n{'='*70}")
print("WEEKLY BREAKDOWN")
print(f"{'='*70}")
for week_name, _, _, _, _ in weeks:
    print(f"\n  {week_name}:")
    sub_w = df[df['Week'] == week_name]
    for strat_name, wh, wa, ws in strategies:
        sub = sub_w[sub_w['Strategy'] == strat_name]
        if len(sub) > 0:
            print(f"    {strat_name:25s}: {sub['Return'].mean()*100:+.2f}% ±{sub['Return'].std()*100:.2f}%")

print(f"\nDone!")
