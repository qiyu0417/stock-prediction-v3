"""
组员模型 + V6管线 vs V5 vs V6 月度对比
用组员的单模型选股 + 我们的V6后处理
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
from collections import Counter
import gc

from config_v5 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from risk_filter import compute_risk_scores, apply_risk_filter
from market_regime import compute_market_regime
from quality_filter import (
    bounce_confirm, compute_quality_score,
    confidence_weighted_allocate, volatility_filter
)

TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
V3_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v2_ensemble')
V5_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v5_ensemble')
TM_DIR = "C:/Users/73065/Desktop/gupiao-3-model/gupiao-3-model/model/v3_disk"
TM_CODE = "C:/Users/73065/Desktop/gupiao-3-model/gupiao-3-model/code/src"

MONTHS = {
    '2026-01': ('2025-12-31', ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']),
    '2026-02': ('2026-01-27', ['2026-02-02', '2026-02-03', '2026-02-04', '2026-02-05', '2026-02-06']),
    '2026-03': ('2026-02-27', ['2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06']),
    '2026-04': ('2026-03-31', ['2026-04-01', '2026-04-02', '2026-04-03', '2026-04-07', '2026-04-08']),
    '2026-05': ('2026-04-30', ['2026-05-04', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08']),
}


# Import teammate's code
sys.path.insert(0, TM_CODE)
from train_v3 import ImprovedStockTransformer, add_cross_sectional_features, fillna_by_stock
from train import _build_label_and_clean


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_v5(df, stockid2idx, winsor_bounds, scaler):
    fe = feature_engineer_func_map[FEATURE_NUM]
    fcols = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    processed = pd.concat([fe(g) for g in tqdm(groups, desc='  V5特征', leave=False)]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])
    for col, (lo, hi) in winsor_bounds.items():
        if col in processed.columns:
            processed[col] = processed[col].clip(lo, hi)
    common = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[common] = scaler.transform(processed[common])
    return processed, common


def preprocess_teammate(df, tm_cfg, stockid2idx):
    """Replicate teammate's preprocessing: feature selection + CS_ features + scaler"""
    fe = feature_engineer_func_map[FEATURE_NUM]
    fc_orig = [c for c in feature_cloums_map[FEATURE_NUM] if c != 'instrument']
    tm_feat = [f for f in tm_cfg['feature_list'] if not f.startswith('CS_')]
    cs_base = [f for f in tm_cfg['feature_list'] if f.startswith('CS_')]

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False)]
    processed = pd.concat([fe(g) for g in tqdm(groups, desc='  TM特征', leave=False)]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    # Fill NA + add CS features
    processed[fc_orig] = processed[fc_orig].replace([np.inf, -np.inf], np.nan)
    processed = fillna_by_stock(processed, fc_orig)
    processed = add_cross_sectional_features(processed, fc_orig)

    # Use teammate's scaler
    scaler = joblib.load(os.path.join(TM_DIR, 'scaler.pkl'))
    all_features = tm_cfg['feature_list']
    common = [c for c in all_features if c in processed.columns]
    if len(common) < len(all_features):
        missing = set(all_features) - set(common)
        for m in missing:
            processed[m] = 0.0
        common = all_features
    processed[common] = scaler.transform(processed[common])
    return processed, common


def build_sequences(data, features, stock_ids, target_date):
    sequences, seq_stock_ids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= target_date)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            sequences.append(hist[features].values.astype(np.float32))
            seq_stock_ids.append(sid)
    return np.asarray(sequences, dtype=np.float32) if sequences else np.array([]), seq_stock_ids


def load_v5_models(fdim, nstocks, device):
    with open(os.path.join(V5_DIR, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    models = []
    for ec in cfg['expert_configs']:
        name = ec['name']
        path = os.path.join(V5_DIR, f'expert_{name}.pth')
        if not os.path.exists(path):
            continue
        if ec['type'] == 'transformer':
            m = StockTransformerExpert(fdim, ec, nstocks)
        elif ec['type'] == 'conv':
            m = ConvStockExpert(fdim, ec, nstocks)
        else:
            continue
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        models.append(m)
    return models, [1.0 / len(models)] * len(models)


def load_teammate_model(fdim, nstocks, device):
    """Load the teammate's single ImprovedStockTransformer"""
    tm_cfg = {
        'sequence_length': SEQUENCE_LENGTH, 'd_model': 256,
        'nhead': 4, 'num_layers': 3, 'dim_feedforward': 512, 'dropout': 0.12,
    }
    model = ImprovedStockTransformer(input_dim=fdim, config=tm_cfg, num_stocks=nstocks).to(device)
    state = torch.load(os.path.join(TM_DIR, 'best_model.pth'), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def v5_ensemble_predict(experts, weights, x, device, seq_ids, risk_scores, market_stress,
                        max_risk_score=85, min_positions=3, max_positions=5):
    NUM_ROUNDS = 5
    MC_SPR = 30
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'

    all_top5 = []
    for r in range(NUM_ROUNDS):
        torch.manual_seed(42 + r * 100)
        np.random.seed(42 + r * 100)
        rnd_scores = []
        for expert in experts:
            expert.train()
            mc = []
            with torch.no_grad():
                for _ in range(MC_SPR):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x.size(1) <= chunk_size:
                            s = expert(x).squeeze(0)
                        else:
                            cs = []
                            for start in range(0, x.size(1), chunk_size):
                                end = min(start + chunk_size, x.size(1))
                                cs.append(expert(x[:, start:end].contiguous()).squeeze(0))
                            s = torch.cat(cs, dim=0)
                    mc.append(s)
            rnd_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())
        fused = np.zeros(len(rnd_scores[0]))
        for w, sc in zip(weights, rnd_scores):
            fused += w * sc
        sel, _ = apply_risk_filter(fused, seq_ids, risk_scores, market_stress,
                                   max_risk_score=max_risk_score, min_positions=min_positions,
                                   max_positions=max_positions)
        all_top5.extend(sel)

    vc = Counter(all_top5)
    consensus = [s for s, c in vc.most_common() if c >= 3]
    if len(consensus) < 1:
        consensus = [s for s, _ in vc.most_common(3)]
    if len(consensus) > 5:
        consensus = consensus[:5]
    weights = [1.0 / len(consensus)] * len(consensus) if consensus else []
    return consensus, weights


def v6_postprocess(raw_scores, seq_ids, processed_data, all_stocks, reference_date):
    """V6后处理: 4D空仓 + 反弹确认 + 质量加分 + t=0.3/30%不等权 + 波动率过滤"""
    score_map = {sid: float(raw_scores[i]) for i, sid in enumerate(seq_ids) if i < len(raw_scores)}

    regime = compute_market_regime(processed_data, [], all_stocks, pd.to_datetime(reference_date))
    if regime.get('skip_trading', False):
        return [], [], regime

    filtered_ids = volatility_filter(processed_data, seq_ids, reference_date, top_pct=0.95)
    if len(filtered_ids) < 3:
        filtered_ids = seq_ids[:10]

    confirmed = bounce_confirm(processed_data, filtered_ids, reference_date)
    for sid in filtered_ids:
        if sid not in confirmed:
            score_map[sid] = score_map.get(sid, 0) * 0.92

    quality = compute_quality_score(processed_data, filtered_ids, reference_date)
    for sid in filtered_ids:
        if sid in score_map and sid in quality:
            score_map[sid] += (quality[sid] - 0.5) * 0.05

    sorted_stocks = sorted(filtered_ids, key=lambda s: score_map.get(s, -999), reverse=True)
    selected, alloc_weights = confidence_weighted_allocate(
        score_map, sorted_stocks, {}, max_positions=5,
        temperature=0.3, max_single=0.30, use_sigma=False)

    return selected, alloc_weights, regime


def _mc_raw_scores(experts, weights, x, device):
    """Get raw MC dropout scores without any post-processing"""
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'
    MC_SPR = 30
    NUM_ROUNDS = 5
    all_fused = []
    for r in range(NUM_ROUNDS):
        torch.manual_seed(42 + r * 100)
        np.random.seed(42 + r * 100)
        rnd_scores = []
        for expert in experts:
            expert.train()
            mc = []
            with torch.no_grad():
                for _ in range(MC_SPR):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x.size(1) <= chunk_size:
                            s = expert(x).squeeze(0)
                        else:
                            cs = []
                            for start in range(0, x.size(1), chunk_size):
                                end = min(start + chunk_size, x.size(1))
                                cs.append(expert(x[:, start:end].contiguous()).squeeze(0))
                            s = torch.cat(cs, dim=0)
                    mc.append(s)
            rnd_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())
        fused = np.zeros(len(rnd_scores[0]))
        for w, sc in zip(weights, rnd_scores):
            fused += w * sc
        all_fused.append(fused)
    return np.mean(all_fused, axis=0)


def calc_week_return(stock_ids, weights, full_data, week_dates):
    week_data = full_data[full_data['日期'].isin(pd.to_datetime(week_dates))]
    filtered = week_data[week_data['股票代码'].isin(stock_ids)]
    if filtered.empty:
        return 0.0
    total = 0.0
    for sid, w in zip(stock_ids, weights):
        stock_w = filtered[filtered['股票代码'] == sid].sort_values('日期')
        if len(stock_w) >= 2:
            ret = (stock_w.iloc[-1]['开盘'] - stock_w.iloc[0]['开盘']) / stock_w.iloc[0]['开盘']
            total += w * ret
    return total


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}\n")

    full_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    # Load configs
    scaler_v5 = joblib.load(os.path.join(V5_DIR, 'scaler.pkl'))
    with open(os.path.join(V5_DIR, 'winsor_bounds.json')) as f:
        winsor_bounds = json.load(f)
    with open(os.path.join(TM_DIR, 'config.json'), encoding='utf-8') as f:
        tm_cfg = json.load(f)

    all_stocks = sorted(full_df['股票代码'].unique())
    fdim = len(feature_cloums_map[FEATURE_NUM])
    nstocks = len(all_stocks)

    print("load models...")
    v5_models, v5_w = load_v5_models(fdim, nstocks, device)
    tm_model = load_teammate_model(len(tm_cfg['feature_list']), nstocks, device)
    print(f"  V5: {len(v5_models)} experts, Teammate: 1 model\n")

    all_results = []

    for month, (cutoff, week_dates) in MONTHS.items():
        print(f"{'='*60}")
        print(f"  {month} | cutoff: {cutoff} | {week_dates[0]} ~ {week_dates[-1]}")
        print(f"{'='*60}")

        train_df = full_df[full_df['日期'] <= cutoff].copy()
        stockid2idx = {s: i for i, s in enumerate(all_stocks)}

        # V5
        pv5, cv5 = preprocess_v5(train_df, stockid2idx, winsor_bounds, scaler_v5)
        sv5, ids_v5 = build_sequences(pv5, cv5, all_stocks, pd.to_datetime(cutoff))
        rv5, stress_v5 = compute_risk_scores(pv5, cv5, all_stocks, ids_v5, train_df['日期'].max())
        rmap_v5 = {sid: rv5.get(sid, 50) for sid in ids_v5}
        xv5 = torch.from_numpy(sv5).unsqueeze(0).to(device)

        # Teammate
        ptm, ctm = preprocess_teammate(train_df, tm_cfg, stockid2idx)
        stm, ids_tm = build_sequences(ptm, ctm, all_stocks, pd.to_datetime(cutoff))
        xtm = torch.from_numpy(stm).unsqueeze(0).to(device)

        # V5 predict
        v5_stocks, v5_wk = v5_ensemble_predict(
            v5_models, v5_w, xv5, device, ids_v5, rmap_v5, stress_v5,
            max_risk_score=85, min_positions=3, max_positions=5)
        r5 = calc_week_return(v5_stocks, v5_wk, full_df, week_dates)

        # V6: get raw scores from V5 models first
        v5_raw = _mc_raw_scores(v5_models, v5_w, xv5, device)
        v6_stocks, v6_wk, regime_v6 = v6_postprocess(
            v5_raw, ids_v5, pv5, all_stocks, cutoff)
        r6_v5 = calc_week_return(v6_stocks, v6_wk, full_df, week_dates)

        # Teammate raw scores (single model, no MC)
        with torch.no_grad():
            tm_scores = tm_model(xtm).squeeze(0).cpu().numpy()

        # Teammate + V6 pipeline
        tm_v6_stocks, tm_v6_wk, regime_tm = v6_postprocess(
            tm_scores, ids_tm, ptm, all_stocks, cutoff)
        r_tm_v6 = calc_week_return(tm_v6_stocks, tm_v6_wk, full_df, week_dates)

        # Teammate raw top5 equal weight (baseline)
        tm_order = np.argsort(tm_scores)[::-1]
        tm_top5 = [ids_tm[i] for i in tm_order[:5] if i < len(ids_tm)]
        tm_wk = [0.2] * len(tm_top5)
        r_tm_raw = calc_week_return(tm_top5, tm_wk, full_df, week_dates)

        print(f"  V5:  {v5_stocks} eq → {r5:+.4%}")
        print(f"  V6:  {v6_stocks} {[f'{w:.1%}' for w in v6_wk] if v6_wk else '空仓'} → {r6_v5:+.4%}")
        print(f"  TM:  {tm_top5} eq → {r_tm_raw:+.4%}")
        tm_label = f"{tm_v6_stocks} {[f'{w:.1%}' for w in tm_v6_wk]}" if tm_v6_wk else "空仓"
        print(f"  TM+V6: {tm_label} → {r_tm_v6:+.4%} | {regime_tm['regime']} ({regime_tm['composite']:.2f})")
        print()

        all_results.append({
            'month': month, 'r5': r5, 'r6_v5': r6_v5,
            'r_tm_raw': r_tm_raw, 'r_tm_v6': r_tm_v6,
            'v5_stocks': v5_stocks, 'v6_stocks': v6_stocks,
            'tm_stocks': tm_top5, 'tm_v6_stocks': tm_v6_stocks,
        })

        del train_df, pv5, ptm; gc.collect()

    # Summary
    print("=" * 80)
    print("  组员模型 vs 我们的模型 月度对比")
    print("=" * 80)
    print(f"  {'Month':<8} {'V5':>8} {'V6':>8} {'TM':>8} {'TM+V6':>8}")
    print("  " + "-" * 45)
    tv5, tv6, ttm, ttm6 = 0, 0, 0, 0
    for r in all_results:
        tv5 += r['r5']; tv6 += r['r6_v5']; ttm += r['r_tm_raw']; ttm6 += r['r_tm_v6']
        print(f"  {r['month']:<8} {r['r5']:>+8.4%} {r['r6_v5']:>+8.4%} {r['r_tm_raw']:>+8.4%} {r['r_tm_v6']:>+8.4%}")
    print("  " + "-" * 45)
    print(f"  {'累计':<8} {tv5:>+8.4%} {tv6:>+8.4%} {ttm:>+8.4%} {ttm6:>+8.4%}")
    print(f"  {'平均':<8} {tv5/5:>+8.4%} {tv6/5:>+8.4%} {ttm/5:>+8.4%} {ttm6/5:>+8.4%}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
