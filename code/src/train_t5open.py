"""
Corrected label: (open_T+5 - open_T+1) / open_T+1
Matches competition formula exactly — previously used close_T+5 instead of open_T+5
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, torch, joblib
from torch.utils.data import DataLoader

from config_stock_emb_8 import *
from train import (create_ranking_dataset_vectorized, RankingDataset, set_seed,
                   calculate_ranking_metrics, _ALPHA_158_COLS, _TECH_39_ONLY,
                   feature_cloums_map, feature_engineer_func_map)
from ensemble_models import StockTransformerExpert, ConvStockExpert
from train_stock_emb_8_loss import preprocess_with_winsor, collate_fn, _make_criterion, train_expert


def build_correct_label(df):
    """Build label: (open_T+5 - open_T+1) / open_T+1 — exact competition formula"""
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df['_open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['_open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df = df[df['_open_t1'] > 1e-4]
    df['label'] = (df['_open_t5'] - df['_open_t1']) / (df['_open_t1'] + 1e-12)
    df = df.dropna(subset=['label'])
    df.drop(columns=['_open_t1', '_open_t5'], inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--features', type=str, default='full', choices=['full', 'alpha'])
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    suffix = '_alpha' if args.features == 'alpha' else ''
    print(f"T5-Open Training [{args.features}] | Loss: {loss_type} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', f'stock_emb_8_t5open{suffix}')
    os.makedirs(output_dir, exist_ok=True)

    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # Feature engineering WITHOUT label (we build corrected label after)
    print("Feature engineering...")
    fe_func = feature_engineer_func_map[FEATURE_NUM]
    df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    min_rows = SEQUENCE_LENGTH + 10
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= min_rows]
    processed = pd.concat([fe_func(g) for g in groups]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    # Build CORRECTED label
    processed = build_correct_label(processed)

    # Feature columns
    feature_columns = feature_cloums_map[FEATURE_NUM]
    if args.features == 'alpha':
        feature_columns = [f for f in _ALPHA_158_COLS if f in feature_columns]
    n_feats = len(feature_columns)
    print(f"  Features: {n_feats}")

    # Winsorize + scale
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=feature_columns)
    from train_stock_emb_8_loss import winsorize_features
    processed, winsor_bounds = winsorize_features(processed, feature_columns, 0.01, 0.99)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    processed[feature_columns] = scaler.fit_transform(processed[feature_columns])

    del full_df; gc.collect()

    # Build dataset
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, feature_columns, SEQUENCE_LENGTH)
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del processed, train_tgt, train_rel, train_stk; gc.collect()

    # Save config
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    with open(os.path.join(output_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f)
    with open(os.path.join(output_dir, 'feature_columns.json'), 'w') as f:
        json.dump(feature_columns, f)

    config_data = {
        'feature_dim': n_feats, 'num_stocks': num_stocks,
        'stock_embed_dim': STOCK_EMBED_DIM, 'expert_configs': EXPERT_CONFIGS,
        't5open': True,
    }
    with open(os.path.join(output_dir, 'ensemble_config.json'), 'w') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    all_results = {}
    for exp_cfg in EXPERT_CONFIGS:
        name = exp_cfg['name']
        model_path = os.path.join(output_dir, f'expert_{name}.pth')
        if os.path.exists(model_path):
            print(f"\n  SKIP {name}: already trained")
            all_results[name] = 0.0
            continue

        print(f"\n{'='*50}")
        print(f"Training: {name} (T5-Open)")
        model = StockTransformerExpert(n_feats, exp_cfg, num_stocks) if exp_cfg['type'] == 'transformer' \
                else ConvStockExpert(n_feats, exp_cfg, num_stocks)
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
        model.to(device)

        best_score = train_expert(model, exp_cfg, dataset, device, name, loss_type, num_epochs)
        all_results[name] = best_score
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone! T5-Open [{args.features}]")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
