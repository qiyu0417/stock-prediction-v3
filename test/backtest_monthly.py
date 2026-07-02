"""
月度回测: Stock Emb vs V7, 2026年1-5月
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm

from config_v5 import SEQUENCE_LENGTH, MAX_STOCKS_PER_CHUNK, USE_AMP, MC_SAMPLES
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, _build_label_and_clean
from market_regime import compute_market_regime
from quality_filter import (
    bounce_confirm, compute_quality_score, volatility_filter, equal_weight_allocate
)

ROOT = os.path.join(os.path.dirname(__file__), '..')
TRAIN = os.path.join(ROOT, 'data', 'train.csv')
V7_DIR = os.path.join(ROOT, 'model', 'v7_ensemble')
STOCK_DIR = os.path.join(ROOT, 'model', 'stock_emb_ensemble')


def load_data():
    train = pd.read_csv(TRAIN, dtype={'股票代码': str})
    train['股票代码'] = train['股票代码'].str.zfill(6)
    train['日期'] = pd.to_datetime(train['日期'].str.replace(' 00:00:00', ''), format='mixed')
    train = train.sort_values(['股票代码', '日期']).reset_index(drop=True)
    return train


def prepare_features(df, winsor, scaler):
    fe = engineer_features_158plus39
    fc = feature_cloums_map['158+39']

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    p = pd.concat([fe(g) for g in tqdm(groups, desc='FE', leave=False)]).reset_index(drop=True)
    p['日期'] = pd.to_datetime(p['日期'].astype(str).str.replace(' 00:00:00', ''), format='mixed')

    stock_ids = sorted(p['股票代码'].unique())
    sid2idx = {s: i for i, s in enumerate(stock_ids)}
    p['instrument'] = p['股票代码'].map(sid2idx)

    for col, (lo, hi) in winsor.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    common = [c for c in scaler.feature_names_in_ if c in p.columns]
    p[common] = scaler.transform(p[common])
    return p, common


def load_models(model_dir, feature_dim, num_stocks, device):
    with open(os.path.join(model_dir, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        if ec['type'] == 'transformer':
            m = StockTransformerExpert(feature_dim, ec, num_stocks)
        else:
            m = ConvStockExpert(feature_dim, ec, num_stocks)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        models.append(m)
    return models, [1.0 / len(models)] * len(models)


def mc_predict(models, weights, x, device):
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'
    all_fused = []
    for r in range(5):
        torch.manual_seed(42 + r * 100)
        np.random.seed(42 + r * 100)
        rnd_scores = []
        for model in models:
            model.train()
            mc = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x.size(1) <= chunk_size:
                            s = model(x).squeeze(0)
                        else:
                            s = torch.cat([model(x[:, i:i+chunk_size]).squeeze(0)
                                         for i in range(0, x.size(1), chunk_size)], dim=0)
                    mc.append(s.cpu().numpy())
            rnd_scores.append(np.mean(mc, axis=0))
        fused = np.zeros_like(rnd_scores[0])
        for w, s in zip(weights, rnd_scores):
            fused += w * s
        all_fused.append(fused)
    return np.mean(all_fused, axis=0)


def monthly_predict(model_dir, model_label, data, pdata, features, device):
    print(f"\n{'='*60}")
    print(f"Monthly Backtest: {model_label}")
    print(f"Model: {model_dir}")

    with open(os.path.join(model_dir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    fdim = cfg.get('feature_dim', len(features))
    nstocks = cfg.get('num_stocks', 300)

    models, weights = load_models(model_dir, fdim, nstocks, device)
    print(f"  loaded {len(models)} experts")

    stock_ids = sorted(data['股票代码'].unique())
    all_dates = sorted(data['日期'].dt.strftime('%Y-%m-%d').unique())
    all_timestamps = sorted(data['日期'].unique())

    monthly_results = {}

    for month in range(1, 6):
        year = 2026
        # Find last trading day of previous month
        prev_month = month - 1
        prev_year = year
        if prev_month == 0:
            prev_month = 12
            prev_year = 2025

        # Find cutoff: last day of prev_month in data
        prev_dates = [d for d in all_timestamps if d.year == prev_year and d.month == prev_month]
        if not prev_dates:
            print(f"  {year}-{month:02d}: no data for previous month, skip")
            continue
        cutoff_dt = max(prev_dates)

        # Find T+1 and T+5 (first 5 trading days of the month)
        month_dates = [d for d in all_timestamps if d.year == year and d.month == month]
        if len(month_dates) < 5:
            print(f"  {year}-{month:02d}: only {len(month_dates)} trading days, skip")
            continue

        t1_dt = month_dates[0]
        t5_dt = month_dates[4]  # 5th trading day

        label_w = f"{year}-{month:02d}"
        cutoff = cutoff_dt.strftime('%Y-%m-%d')
        t1_open = t1_dt.strftime('%Y-%m-%d')
        t5_open = t5_dt.strftime('%Y-%m-%d')

        # Build input batch
        seqs, sids = [], []
        for sid in stock_ids:
            hist = pdata[(pdata['股票代码'] == sid) & (pdata['日期'] <= cutoff_dt)].sort_values('日期').tail(SEQUENCE_LENGTH)
            if len(hist) == SEQUENCE_LENGTH:
                seqs.append(hist[features].values.astype(np.float32))
                sids.append(sid)

        if not seqs:
            print(f"  {label_w}: no sequences")
            continue
        x = torch.FloatTensor(np.stack(seqs)).unsqueeze(0).to(device)

        raw = mc_predict(models, weights, x, device)
        raw_scores = {sid: float(raw[i]) for i, sid in enumerate(sids)}

        # Market regime
        regime = compute_market_regime(data, features, sids, cutoff_dt)
        skip = regime.get('skip_trading', False)

        if skip:
            print(f"  {label_w}: SKIP (market regime) | cutoff={cutoff}, t1={t1_open}, t5={t5_open}")
            monthly_results[label_w] = {'return': 0, 'skip': True, 'stocks': []}
            continue

        # Post-processing
        kept = volatility_filter(data, sids, cutoff_dt, top_pct=0.95)
        confirmed = bounce_confirm(data, kept, cutoff_dt, threshold=0.008)
        quality = compute_quality_score(data, kept, cutoff_dt)

        adjusted = {}
        for sid in kept:
            s = raw_scores.get(sid, 0)
            if sid not in confirmed:
                s *= 0.92
            q = quality.get(sid, 0.5)
            s += (q - 0.5) * 0.05
            adjusted[sid] = s

        ranked = sorted(adjusted.items(), key=lambda x: -x[1])
        selected, weights_alloc = equal_weight_allocate([s for s, _ in ranked], 5)

        # Calculate return
        ret = 0
        if selected:
            t1_map, t5_map = {}, {}
            for sid in selected:
                sd = data[(data['股票代码'] == sid) & (data['日期'].dt.strftime('%Y-%m-%d') == t1_open)]
                if len(sd) > 0:
                    t1_map[sid] = float(sd['开盘'].iloc[0])
                sd5 = data[(data['股票代码'] == sid) & (data['日期'].dt.strftime('%Y-%m-%d') == t5_open)]
                if len(sd5) > 0:
                    t5_map[sid] = float(sd5['开盘'].iloc[0])

            stock_rets = []
            for sid in selected:
                if sid in t1_map and sid in t5_map and t1_map[sid] > 0:
                    r = (t5_map[sid] - t1_map[sid]) / t1_map[sid]
                    stock_rets.append(r)
            if stock_rets:
                ret = np.mean(stock_rets)

        print(f"  {label_w}: Top5={selected} | return={ret:+.4f} ({ret*100:+.2f}%) | "
              f"cutoff={cutoff}, t1={t1_open}, t5={t5_open}")
        monthly_results[label_w] = {'return': ret, 'skip': False, 'stocks': selected}

        del x
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return monthly_results


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    data = load_data()
    all_dates = sorted(data['日期'].unique())
    print(f"Data: {data['日期'].min().date()} ~ {data['日期'].max().date()}")
    print(f"Trading days per month:")
    for m in range(1, 6):
        month_dates = [d for d in all_dates if d.year == 2026 and d.month == m]
        if month_dates:
            print(f"  2026-{m:02d}: {len(month_dates)} days, first={month_dates[0].date()}, last={month_dates[-1].date()}")

    # V7 - 197 features, no industry
    v7_winsor = json.load(open(os.path.join(V7_DIR, 'winsor_bounds.json')))
    v7_scaler = joblib.load(os.path.join(V7_DIR, 'scaler.pkl'))
    pdata_v7, features_v7 = prepare_features(data.copy(), v7_winsor, v7_scaler)

    # Stock Emb - 197 features, no industry
    stock_winsor = json.load(open(os.path.join(STOCK_DIR, 'winsor_bounds.json')))
    stock_scaler = joblib.load(os.path.join(STOCK_DIR, 'scaler.pkl'))
    pdata_stock, features_stock = prepare_features(data.copy(), stock_winsor, stock_scaler)

    r7 = monthly_predict(V7_DIR, "V7 (DeepSleep V1)", data, pdata_v7, features_v7, device)
    rs = monthly_predict(STOCK_DIR, "Stock Emb", data, pdata_stock, features_stock, device)

    # Summary
    print(f"\n{'='*60}")
    print("MONTHLY COMPARISON (Jan-May 2026)")
    print(f"{'Month':<10} | {'V7':>10} | {'Stock Emb':>12} | {'Diff':>10}")
    print("-" * 52)
    all_keys = sorted(set(list(r7.keys()) + list(rs.keys())))
    for k in all_keys:
        v7_r = r7.get(k, {}).get('return', 0) * 100
        st_r = rs.get(k, {}).get('return', 0) * 100
        diff = st_r - v7_r
        print(f"{k:<10} | {v7_r:+9.2f}% | {st_r:+11.2f}% | {diff:+9.2f}%")

    # Cumulative
    v7_cum = 1.0
    st_cum = 1.0
    for k in all_keys:
        v7_cum *= (1 + r7.get(k, {}).get('return', 0))
        st_cum *= (1 + rs.get(k, {}).get('return', 0))
    v7_total = (v7_cum - 1) * 100
    st_total = (st_cum - 1) * 100
    print("-" * 52)
    print(f"{'Cumulative':<10} | {v7_total:+9.2f}% | {st_total:+11.2f}% | {st_total-v7_total:+9.2f}%")


if __name__ == '__main__':
    main()
