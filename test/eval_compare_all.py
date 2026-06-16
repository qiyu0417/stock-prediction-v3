"""
对比评测: dim=16+Hybrid vs dim=8 Hybrid vs dim=16 k=3 vs dim=8 k=3
"""
import os, sys, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, joblib

from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from ensemble_models import StockTransformerExpert, ConvStockExpert
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC_PASSES = 5
SEQUENCE_LENGTH = 60

MODELS = {
    'dim=16+Hybrid': 'model/stock_emb_16_dim16_hybrid',
    'dim=16 k=3 T=0.5': 'model/stock_emb_16_dim16_listmle_k3_t0.5',
    'dim=8 Hybrid': 'model/stock_emb_8_hybrid',
    'dim=8 k=3 T=0.5': 'model/stock_emb_8_listmle_k3_t0.5',
}


def preprocess_eval(df, stockid2idx, scaler, winsor_bounds):
    from config_stock_emb_8 import FEATURE_NUM
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    min_rows = SEQUENCE_LENGTH + 10
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False)
              if len(g) >= min_rows]

    if len(groups) == 0:
        return None, None, None, None

    processed = pd.concat([feature_engineer(g) for g in groups]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=True)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=feature_columns)

    for col, (lo, hi) in winsor_bounds.items():
        if col in processed.columns:
            processed[col] = processed[col].clip(lo, hi)

    processed[feature_columns] = scaler.transform(processed[feature_columns])
    return processed, feature_columns


def load_experts(model_dir, feature_dim, num_stocks, device):
    with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
        cfg = json.load(f)
    embed_dim = cfg.get('stock_embed_dim', 8)
    expert_configs = cfg['expert_configs']
    models = []
    for ec in expert_configs:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        ec_copy = dict(ec)
        ec_copy['stock_embed_dim'] = embed_dim
        if ec['type'] == 'transformer':
            model = StockTransformerExpert(feature_dim, ec_copy, num_stocks)
        else:
            model = ConvStockExpert(feature_dim, ec_copy, num_stocks)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.train()
        models.append(model)
    return models


def predict_and_score(models, processed, features, raw_df, pred_date_str, device):
    """Predict raw scores then apply post-processing"""
    n_feats = len(features)
    seq_len = SEQUENCE_LENGTH

    # Get stocks available up to pred_date
    pred_dt = pd.to_datetime(pred_date_str)
    hist = processed[processed['日期'] <= pred_date_str]
    available = hist['股票代码'].unique()
    stock_ids = sorted(available)
    n_stocks = len(stock_ids)
    if n_stocks < 5:
        return None

    # Build sequences
    sequences = np.zeros((1, n_stocks, seq_len, n_feats), dtype=np.float32)
    valid_mask = np.zeros(n_stocks, dtype=bool)
    for i, sid in enumerate(stock_ids):
        stock_data = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(stock_data) >= seq_len:
            sequences[0, i] = stock_data[features].values[-seq_len:].astype(np.float32)
            valid_mask[i] = True

    seq_t = torch.FloatTensor(sequences).to(device)

    # MC Dropout ensemble
    all_scores = []
    for _ in range(MC_PASSES):
        pass_scores = []
        for model in models:
            with torch.no_grad():
                pred = model(seq_t)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred[0].cpu().numpy()
            pass_scores.append(pred)
        all_scores.append(np.mean(pass_scores, axis=0))
    mc_scores = np.mean(all_scores, axis=0)

    # Build dict: stock_id -> raw_score
    raw_scores = {}
    for i, sid in enumerate(stock_ids):
        if valid_mask[i]:
            raw_scores[sid] = float(mc_scores[i])
        else:
            raw_scores[sid] = -float('inf')

    # Post-processing pipeline
    data = raw_df[raw_df['日期'] <= pred_date_str]
    filtered_ids = volatility_filter(data, stock_ids, pred_date_str, top_pct=0.95)
    bounce_flags = bounce_confirm(data, filtered_ids, pred_date_str)
    quality_scores = compute_quality_score(data, filtered_ids, pred_date_str)

    # Apply bounce penalty and quality bonus
    final_scores = {}
    for sid in filtered_ids:
        score = raw_scores.get(sid, -float('inf'))
        if sid not in bounce_flags:  # bounce_confirm returns set of confirmed stocks
            score *= 0.92
        quality = quality_scores.get(sid, 0.5)
        score += (quality - 0.5) * 0.05
        final_scores[sid] = score

    # Select top 5
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top5_ids = [sid for sid, _ in ranked[:5]]
    selected, weights = equal_weight_allocate(top5_ids)

    return list(zip(selected, weights))


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")
    set_seed(42)

    # Load data
    train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
    train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6)
    train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')

    test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
    test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
    test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['股票代码', '日期'], keep='last')
    test_dates = sorted(test_df['日期'].unique())

    # Need raw (unscaled) data for post-processing
    raw_data = full_df.copy()

    # Prediction targets
    pred_dates = ['2026-06-01', '2026-06-08']

    all_results = {}

    for model_name, model_dir in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        if not os.path.exists(model_dir):
            print(f"  SKIP")
            continue

        with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
            cfg = json.load(f)
        feature_dim = cfg['feature_dim']
        num_stocks = cfg['num_stocks']

        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        with open(os.path.join(model_dir, 'winsor_bounds.json'), 'r') as f:
            winsor_bounds = json.load(f)

        all_stock_ids = sorted(full_df['股票代码'].unique())
        stockid2idx = {s: i for i, s in enumerate(all_stock_ids)}

        print("  Preprocessing...")
        result = preprocess_eval(full_df, stockid2idx, scaler, winsor_bounds)
        if result[0] is None:
            print("  FAILED")
            continue
        processed, features = result
        print(f"  Features: {len(features)}")

        models = load_experts(model_dir, feature_dim, num_stocks, device)
        print(f"  Experts: {len(models)}")

        model_returns = []
        for pred_date_str in pred_dates:
            result = predict_and_score(models, processed, features, raw_data, pred_date_str, device)
            if result is None:
                print(f"  {pred_date_str}: NO PREDICTION")
                continue

            picks = result
            stock_ids = [p[0] for p in picks]

            # Calculate return: T+1 open → min(T+5, last available) open
            t1_date = test_dates[0]
            for d in test_dates:
                if d >= pd.to_datetime(pred_date_str):
                    t1_date = d
                    break

            t5_idx = min(4, len([d for d in test_dates if d >= pd.to_datetime(pred_date_str)]) - 1)
            t5_dates_in_range = [d for d in test_dates if d >= pd.to_datetime(pred_date_str)]
            t5_date = t5_dates_in_range[t5_idx] if t5_dates_in_range else test_dates[-1]

            returns = []
            for sid, weight in picks:
                t1_data = test_df[(test_df['股票代码'] == sid) & (test_df['日期'] == t1_date)]
                t5_data = test_df[(test_df['股票代码'] == sid) & (test_df['日期'] == t5_date)]
                if len(t1_data) == 0 or len(t5_data) == 0:
                    ret = 0.0
                else:
                    ret = (float(t5_data.iloc[0]['开盘']) - float(t1_data.iloc[0]['开盘'])) / float(t1_data.iloc[0]['开盘'])
                returns.append(ret * weight)

            week_ret = sum(returns)
            model_returns.append(week_ret)
            print(f"  {pred_date_str} -> {str(t5_date.date())}: {stock_ids} | {week_ret*100:+.2f}%")

        if model_returns:
            avg = np.mean(model_returns)
            all_results[model_name] = {'returns': model_returns, 'avg': avg}
            w1 = model_returns[0] if len(model_returns) > 0 else 0
            w2 = model_returns[1] if len(model_returns) > 1 else 0
            print(f"  => W1: {w1*100:+.2f}%  W2: {w2*100:+.2f}%  Avg: {avg*100:+.2f}%")

        del models, processed
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Model':<25} {'Jun W1':>10} {'Jun W2':>10} {'Avg':>10}")
    print("-" * 60)
    for name in MODELS:
        if name in all_results:
            r = all_results[name]['returns']
            w1 = f"{r[0]*100:+.2f}%" if len(r) > 0 else "N/A"
            w2 = f"{r[1]*100:+.2f}%" if len(r) > 1 else "N/A"
            avg = f"{all_results[name]['avg']*100:+.2f}%"
            print(f"{name:<25} {w1:>10} {w2:>10} {avg:>10}")

    print("\nDone!")


if __name__ == '__main__':
    main()
