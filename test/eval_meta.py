"""
Evaluate meta-model weighted ensemble: MC=20 + 5 seeds on 3 June weeks.
Compares Hybrid-only, Alpha158-only, simple avg, fixed optimal w, and meta-model.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate
from market_regime import compute_market_regime

MC = 20
SEQ = 60
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

# ── Meta-model class (same as train_meta.py) ──
META_FEATURE_COLS = [
    'trend_score', 'breadth_score', 'accel_decline_score',
    'volatility_score', 'composite',
    'ret_5d', 'ret_10d', 'ret_20d',
    'market_up_ratio_5d', 'market_return_5d',
    'market_volatility_5d', 'consecutive_downs',
]


class MarketRegimeMLP(nn.Module):
    def __init__(self, input_dim=len(META_FEATURE_COLS), hidden=8, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Compute market features at reference date ──
def compute_market_features(ref_date, processed_raw, all_sids):
    regime = compute_market_regime(processed_raw, [], all_sids, ref_date)

    daily = processed_raw[processed_raw['日期'] <= ref_date].tail(300 * len(all_sids))
    by_date = daily.groupby('日期')

    ret_5d = ret_10d = ret_20d = 0.0
    market_up_ratio = 0.5
    market_return_5d = 0.0
    market_vol_5d = 0.0

    if '涨跌幅' in daily.columns:
        market_return_daily = by_date['涨跌幅'].mean()
        recent = market_return_daily.tail(30)
        if len(recent) >= 5:
            ret_5d = float(recent.tail(5).sum())
            market_up_ratio = float((recent.tail(5) > 0).mean())
            market_return_5d = float(recent.tail(5).mean())
            market_vol_5d = float(recent.tail(5).std())
        if len(recent) >= 10:
            ret_10d = float(recent.tail(10).sum())
        if len(recent) >= 20:
            ret_20d = float(recent.tail(20).sum())

    features = {
        'trend_score': regime.get('trend_score', 0.5),
        'breadth_score': regime.get('breadth_score', 0.5),
        'accel_decline_score': regime.get('accel_decline_score', 0.0),
        'volatility_score': regime.get('volatility_score', 0.0),
        'composite': regime.get('composite', 0.5),
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'ret_20d': ret_20d,
        'market_up_ratio_5d': market_up_ratio,
        'market_return_5d': market_return_5d,
        'market_volatility_5d': market_vol_5d,
        'consecutive_downs': 1.0 if regime.get('consecutive_downs', False) else 0.0,
    }
    return features


# ── Load meta-model ──
def load_meta_model():
    model_dir = 'model/meta_model'
    with open(os.path.join(model_dir, 'meta_config.json')) as f:
        cfg = json.load(f)
    scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    model = MarketRegimeMLP(input_dim=len(cfg['feature_cols']))
    model.load_state_dict(
        torch.load(os.path.join(model_dir, 'meta_model.pth'),
                   map_location=torch.device('cuda'), weights_only=True))
    model.to(torch.device('cuda'))
    model.eval()
    return model, scaler, cfg


# ── Load ensemble models ──
def load_model(mdir, nf, device):
    import ensemble_models as _em
    _orig_ti = _em.StockTransformerExpert.__init__
    _orig_ci = _em.ConvStockExpert.__init__
    _em.StockTransformerExpert.__init__ = _orig_ti
    _em.ConvStockExpert.__init__ = _orig_ci
    from ensemble_models import StockTransformerExpert, ConvStockExpert

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


# ── MC inference ──
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


# ── Build sequence ──
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


# ── Compute weighted return for a given weight ──
def compute_weighted_return(w, h_raw, a_raw, h_sids, a_sids, h_valid, a_valid,
                            raw_hist, ref_str, t1_date, t5_date, full_df):
    h_map = {s: float(h_raw[i]) for i, s in enumerate(h_sids) if h_valid[i]}
    a_map = {s: float(a_raw[i]) for i, s in enumerate(a_sids) if a_valid[i]}
    common = sorted(set(h_map.keys()) & set(a_map.keys()))
    if len(common) < 10:
        return None, [], {}

    combined = {sid: w * h_map.get(sid, 0) + (1 - w) * a_map.get(sid, 0)
                 for sid in common}

    filt = volatility_filter(raw_hist, list(common), ref_str, top_pct=VP)
    if len(filt) < 5:
        return None, [], {}
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
    sids_top = [s for s, _ in top5]
    _, weights = equal_weight_allocate(sids_top)
    ret = 0.0
    for sid, wgt in zip(sids_top, weights):
        r1 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t1_date)]
        r5 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t5_date)]
        if len(r1) > 0 and len(r5) > 0:
            sr = (float(r5.iloc[0]['开盘']) - float(r1.iloc[0]['开盘'])) / float(r1.iloc[0]['开盘'])
            ret += sr * wgt
    return ret, sids_top, final


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    device = torch.device('cuda')
    print(f"Loading data & models...")

    # Load data
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

    # Preprocess (full feature set for both)
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

    H = load_model('model/stock_emb_8_hybrid', len(fcols_all), device)
    p_h = preprocess('model/stock_emb_8_hybrid')
    A = load_model('model/stock_emb_8_alpha158', len(alpha_f), device)
    p_a = preprocess('model/stock_emb_8_alpha158')

    # Load meta-model
    meta_model, meta_scaler, meta_cfg = load_meta_model()
    fixed_w = 0.45  # mean optimal w from training data

    print(f"  Models loaded. Meta input dim: {meta_cfg['input_dim']}")

    # Evaluation weeks
    weeks = [
        ('W1', pd.to_datetime('2026-05-29'), pd.to_datetime('2026-06-01'),
         pd.to_datetime('2026-06-05'), 5),
        ('W2', pd.to_datetime('2026-06-05'), pd.to_datetime('2026-06-08'),
         pd.to_datetime('2026-06-12'), 5),
        ('W3', pd.to_datetime('2026-06-12'), pd.to_datetime('2026-06-15'),
         pd.to_datetime('2026-06-18'), 4),
    ]

    seeds = [42, 123, 456, 789, 1024]
    strategies = ['Hybrid-only', 'Alpha158-only', 'SimpleAvg(w=0.5)',
                  f'Fixed(w={fixed_w})', 'Meta-model']

    all_results = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n--- Seed={seed} ---")

        for wname, pd_str, t1_date, t5_date, ndays in weeks:
            ref_str = str(pd_str.date())
            ref_date = pd_str

            # Build sequences
            seq_h, sids_h, valid_h = build_seq(p_h, ref_date, fcols_all, len(fcols_all))
            seq_a, sids_a, valid_a = build_seq(p_a, ref_date, alpha_f, len(alpha_f))

            # MC inference
            seq_ht = torch.FloatTensor(seq_h).to(device)
            raw_h = mc_infer(H, seq_ht)
            seq_at = torch.FloatTensor(seq_a).to(device)
            raw_a = mc_infer(A, seq_at)

            raw_hist = processed_raw[processed_raw['日期'] <= ref_date]

            # Compute market features for meta-model
            mkt = compute_market_features(ref_date, processed_raw, all_sids)
            X_meta = np.array([[mkt[c] for c in META_FEATURE_COLS]], dtype=np.float32)
            X_meta_s = meta_scaler.transform(X_meta)
            with torch.no_grad():
                w_meta = float(meta_model(torch.FloatTensor(X_meta_s).to(device)).cpu())
            w_meta = max(0.0, min(1.0, w_meta))

            # Evaluate each strategy
            strategy_weights = {
                'Hybrid-only': 1.0,
                'Alpha158-only': 0.0,
                'SimpleAvg(w=0.5)': 0.5,
                f'Fixed(w={fixed_w})': fixed_w,
                'Meta-model': w_meta,
            }

            for strat, w in strategy_weights.items():
                ret, top5, _ = compute_weighted_return(
                    w, raw_h, raw_a, sids_h, sids_a, valid_h, valid_a,
                    raw_hist, ref_str, t1_date, t5_date, full_df)
                if ret is not None:
                    all_results.append({
                        'Seed': seed, 'Week': wname, 'Strategy': strat,
                        'Return': ret, 'w': w, 'Top5': ','.join(top5),
                    })

            print(f"  {wname}: w_meta={w_meta:.3f} "
                  f"H={strategy_weights['Hybrid-only']} "
                  f"A={strategy_weights['Alpha158-only']}")

    # ── Summary ──
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("RESULTS: 5-seed MC=20 mean returns")
    print(f"{'='*70}")

    # Overall averages
    for strat in strategies:
        sub = df[df['Strategy'] == strat]
        mean_ret = sub['Return'].mean() * 100
        std_ret = sub['Return'].std() * 100
        print(f"  {strat:20s}: {mean_ret:+.2f}% ± {std_ret:.2f}%")

    # Weekly breakdown
    print(f"\n{'='*70}")
    print("WEEKLY BREAKDOWN")
    print(f"{'='*70}")
    for week_name, _, _, _, _ in weeks:
        print(f"\n  {week_name}:")
        sub_w = df[df['Week'] == week_name]
        for strat in strategies:
            sub = sub_w[sub_w['Strategy'] == strat]
            if len(sub) > 0:
                print(f"    {strat:20s}: {sub['Return'].mean()*100:+.2f}% "
                      f"±{sub['Return'].std()*100:.2f}% "
                      f"(w avg={sub['w'].mean():.3f})")

    # Show meta-model w predictions per week
    print(f"\n{'='*70}")
    print("META-MODEL WEIGHTS BY WEEK")
    print(f"{'='*70}")
    for week_name, _, _, _, _ in weeks:
        sub = df[(df['Week'] == week_name) & (df['Strategy'] == 'Meta-model')]
        if len(sub) > 0:
            ws = sub['w'].values
            print(f"  {week_name}: w = {ws.mean():.3f} ± {ws.std():.3f}")

    print(f"\nDone!")
