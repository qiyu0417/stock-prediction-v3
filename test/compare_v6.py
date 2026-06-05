"""
V3 vs V5 vs V6 三方月度对比 (2026年1-5月)
V6: 4D市场状态 + 反弹确认 + 质量加分 + 置信度不等权 + 波动率过滤
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

MONTHS = {
    '2026-01': ('2025-12-31', ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']),
    '2026-02': ('2026-01-27', ['2026-02-02', '2026-02-03', '2026-02-04', '2026-02-05', '2026-02-06']),
    '2026-03': ('2026-02-27', ['2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06']),
    '2026-04': ('2026-03-31', ['2026-04-01', '2026-04-02', '2026-04-03', '2026-04-07', '2026-04-08']),
    '2026-05': ('2026-04-30', ['2026-05-04', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08']),
}


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_v3(df, stockid2idx, scaler):
    fe = feature_engineer_func_map[FEATURE_NUM]
    fcols = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    processed = pd.concat([fe(g) for g in tqdm(groups, desc='  V3特征', leave=False)]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])
    common = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[common] = scaler.transform(processed[common])
    return processed, common


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


def build_sequences(data, features, stock_ids, target_date):
    sequences, seq_stock_ids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= target_date)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            sequences.append(hist[features].values.astype(np.float32))
            seq_stock_ids.append(sid)
    return np.asarray(sequences, dtype=np.float32) if sequences else np.array([]), seq_stock_ids


def load_v3(fdim, nstocks, device):
    models, names = [], ['balanced_v2', 'deep_v2', 'conv_multiscale', 'conv_deep']
    w_raw = [0.1855, 0.1215, 0.1113, 0.0804]
    w_norm = [x / sum(w_raw) for x in w_raw]
    for name in names:
        path = os.path.join(V3_DIR, f'expert_{name}.pth')
        if name.startswith('conv'):
            cfg = {'name': name, 'type': 'conv', 'hidden_channels': 256 if 'multi' in name else 384,
                   'nhead': 4, 'dropout': 0.12 if 'multi' in name else 0.15,
                   'mc_dropout_rate': 0.1 if 'multi' in name else 0.12, 'sd_prob': 0.9 if 'multi' in name else 0.85}
            m = ConvStockExpert(fdim, cfg, nstocks)
        else:
            cfg = {'name': name, 'type': 'transformer',
                   'd_model': 256 if name == 'balanced_v2' else 192, 'nhead': 4,
                   'num_layers': 6 if name == 'balanced_v2' else 8,
                   'dim_feedforward': 512 if name == 'balanced_v2' else 384,
                   'dropout': 0.1, 'mc_dropout_rate': 0.1 if name == 'balanced_v2' else 0.12,
                   'sd_prob': 0.9 if name == 'balanced_v2' else 0.85}
            m = StockTransformerExpert(fdim, cfg, nstocks)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        models.append(m)
    return models, w_norm


def load_v5(fdim, nstocks, device):
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


def v3_ensemble_predict(experts, weights, x, device, seq_ids, risk_scores, market_stress,
                        max_risk_score=85, min_positions=3, max_positions=5):
    """V3/V5: 投票共识 + 风险过滤 + 等权"""
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


def v6_predict(experts, weights, x, device, seq_ids,
               processed_data, all_stocks, reference_date):
    """V6: 4D空仓 + 反弹确认 + 质量加分 + t=0.3/30%上限不等权 + 波动率过滤"""
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
    raw_scores = np.mean(all_fused, axis=0)
    score_map = {sid: float(raw_scores[i]) for i, sid in enumerate(seq_ids) if i < len(raw_scores)}

    # 4D 市场状态
    regime = compute_market_regime(processed_data, [], all_stocks, pd.to_datetime(reference_date))

    # 空仓: risk_off / 广度崩溃 / 连续下跌 / 加速跌+趋势双高
    if regime.get('skip_trading', False):
        return [], []

    # 波动率过滤
    filtered_ids = volatility_filter(processed_data, seq_ids, reference_date, top_pct=0.95)
    if len(filtered_ids) < 3:
        filtered_ids = seq_ids[:10]

    # 反弹确认 (轻惩罚)
    confirmed = bounce_confirm(processed_data, filtered_ids, reference_date)
    for sid in filtered_ids:
        if sid not in confirmed:
            score_map[sid] = score_map.get(sid, 0) * 0.92

    # 质量加分 (轻权重)
    quality = compute_quality_score(processed_data, filtered_ids, reference_date)
    for sid in filtered_ids:
        if sid in score_map and sid in quality:
            q_bonus = quality[sid] - 0.5
            score_map[sid] += q_bonus * 0.05

    # 不等权分配: t=0.3 + 30%上限 (4D负责空仓, σ不重复降仓位)
    sorted_stocks = sorted(filtered_ids, key=lambda s: score_map.get(s, -999), reverse=True)
    selected, alloc_weights = confidence_weighted_allocate(
        score_map, sorted_stocks, regime, max_positions=5,
        temperature=0.3, max_single=0.30, use_sigma=False)

    return selected, alloc_weights


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
    print(f"设备: {device}\n")

    full_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    scaler_v3 = joblib.load(os.path.join(V3_DIR, 'scaler.pkl'))
    scaler_v5 = joblib.load(os.path.join(V5_DIR, 'scaler.pkl'))
    with open(os.path.join(V5_DIR, 'winsor_bounds.json')) as f:
        winsor_bounds = json.load(f)

    all_stocks = sorted(full_df['股票代码'].unique())
    fdim = len(feature_cloums_map[FEATURE_NUM])
    nstocks = len(all_stocks)

    print("加载模型...")
    v3_models, v3_w = load_v3(fdim, nstocks, device)
    v5_models, v5_w = load_v5(fdim, nstocks, device)
    print(f"  V3: {len(v3_models)}, V5/V6: {len(v5_models)} experts\n")

    all_results = []

    for month, (cutoff, week_dates) in MONTHS.items():
        print(f"{'='*60}")
        print(f"  {month} | 截止: {cutoff} | {week_dates[0]} ~ {week_dates[-1]}")
        print(f"{'='*60}")

        train_df = full_df[full_df['日期'] <= cutoff].copy()
        stockid2idx = {s: i for i, s in enumerate(all_stocks)}

        # V3
        pv3, cv3 = preprocess_v3(train_df, stockid2idx, scaler_v3)
        sv3, ids_v3 = build_sequences(pv3, cv3, all_stocks, pd.to_datetime(cutoff))
        rv3, stress_v3 = compute_risk_scores(pv3, cv3, all_stocks, ids_v3, train_df['日期'].max())
        rmap_v3 = {sid: rv3.get(sid, 50) for sid in ids_v3}
        xv3 = torch.from_numpy(sv3).unsqueeze(0).to(device)

        # V5
        pv5, cv5 = preprocess_v5(train_df, stockid2idx, winsor_bounds, scaler_v5)
        sv5, ids_v5 = build_sequences(pv5, cv5, all_stocks, pd.to_datetime(cutoff))
        rv5, stress_v5 = compute_risk_scores(pv5, cv5, all_stocks, ids_v5, train_df['日期'].max())
        rmap_v5 = {sid: rv5.get(sid, 50) for sid in ids_v5}
        xv5 = torch.from_numpy(sv5).unsqueeze(0).to(device)

        # V3 predict
        v3_stocks, v3_wk = v3_ensemble_predict(
            v3_models, v3_w, xv3, device, ids_v3, rmap_v3, stress_v3,
            max_risk_score=85, min_positions=3, max_positions=5)
        r3 = calc_week_return(v3_stocks, v3_wk, full_df, week_dates)

        # V5 predict
        v5_stocks, v5_wk = v3_ensemble_predict(
            v5_models, v5_w, xv5, device, ids_v5, rmap_v5, stress_v5,
            max_risk_score=85, min_positions=3, max_positions=5)
        r5 = calc_week_return(v5_stocks, v5_wk, full_df, week_dates)

        # V6 predict (new pipeline) - compute regime first for diagnostics
        v6_regime = compute_market_regime(pv5, [], all_stocks, pd.to_datetime(cutoff))
        v6_stocks, v6_wk = v6_predict(
            v5_models, v5_w, xv5, device, ids_v5, pv5, all_stocks, cutoff)
        r6 = calc_week_return(v6_stocks, v6_wk, full_df, week_dates)

        print(f"  V3: {v3_stocks} 等权 → {r3:+.4%}")
        print(f"  V5: {v5_stocks} 等权 → {r5:+.4%}")
        v6_label = "空仓" if v6_regime.get('skip_trading') else \
                   f"{v6_stocks} {[f'{w:.1%}' for w in v6_wk] if v6_wk else '无'}"
        print(f"  V6: {v6_label} → {r6:+.4%} | 市场: {v6_regime['regime']} "
              f"(趋势{v6_regime['trend_score']:.1f}/广度{v6_regime['breadth_score']:.1f}/"
              f"加速跌{v6_regime['accel_decline_score']:.1f}/波动{v6_regime['volatility_score']:.1f})")

        d5 = r5 - r3
        d6 = r6 - r5
        f5 = "V5" if d5 > 0.001 else ("V3" if d5 < -0.001 else "=")
        f6 = "V6" if d6 > 0.001 else ("V5" if d6 < -0.001 else "=")
        print(f"  V5 vs V3: {f5} ({d5:+.4%}) | V6 vs V5: {f6} ({d6:+.4%})\n")

        all_results.append({
            'month': month, 'v3': v3_stocks, 'v5': v5_stocks, 'v6': v6_stocks,
            'r3': r3, 'r5': r5, 'r6': r6, 'v6_weights': v6_wk
        })

        del train_df, pv3, pv5; gc.collect()

    # 汇总
    print("=" * 80)
    print("  V3 vs V5 vs V6 月度对比汇总 (2026年1-5月)")
    print("=" * 80)
    print(f"  {'月份':<8} {'V3收益':>8} {'V5收益':>8} {'V6收益':>8} {'V5-V3':>8} {'V6-V5':>8}")
    print("  " + "-" * 50)
    tv3, tv5, tv6 = 0, 0, 0
    w5v3, w6v5 = 0, 0
    for r in all_results:
        tv3 += r['r3']; tv5 += r['r5']; tv6 += r['r6']
        d5 = r['r5'] - r['r3']; d6 = r['r6'] - r['r5']
        if d5 > 0.001: w5v3 += 1
        if d6 > 0.001: w6v5 += 1
        print(f"  {r['month']:<8} {r['r3']:>+8.4%} {r['r5']:>+8.4%} {r['r6']:>+8.4%} {d5:>+8.4%} {d6:>+8.4%}")
    print("  " + "-" * 50)
    print(f"  {'累计':<8} {tv3:>+8.4%} {tv5:>+8.4%} {tv6:>+8.4%} {tv5-tv3:>+8.4%} {tv6-tv5:>+8.4%}")
    print(f"  {'平均':<8} {tv3/5:>+8.4%} {tv5/5:>+8.4%} {tv6/5:>+8.4%}")
    print(f"\n  V5胜V3: {w5v3}/5 月, V6胜V5: {w6v5}/5 月")

    print("\n" + "=" * 80)
    print("  选股详情")
    print("=" * 80)
    for r in all_results:
        print(f"  {r['month']}:")
        print(f"    V3: {r['v3']}")
        print(f"    V5: {r['v5']}")
        v6_info = list(zip(r['v6'], [f'{w:.1%}' for w in r['v6_weights']])) if r['v6_weights'] else []
        print(f"    V6: {v6_info}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
