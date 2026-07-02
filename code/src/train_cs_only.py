"""
Pure cross-sectional model: only CS_ features (25-dim z-score rankings)
Fundamentally different from time-series models — captures relative strength
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, torch, joblib
from torch.utils.data import DataLoader

from config_stock_emb_8 import *
from train import (create_ranking_dataset_vectorized, RankingDataset, set_seed,
                   calculate_ranking_metrics, _ALPHA_158_COLS, feature_cloums_map,
                   feature_engineer_func_map, _build_label_and_clean)
from ensemble_models import StockTransformerExpert, ConvStockExpert
from train_stock_emb_8_loss import collate_fn, _make_criterion, train_expert, winsorize_features
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"CS-only Training | Loss: {loss_type} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_cs_only')
    os.makedirs(output_dir, exist_ok=True)

    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # Feature engineering: full features + CS
    from utils import engineer_features_158plus39, add_cross_sectional_features
    from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean

    print("Engineering features...")
    fe_func = feature_engineer_func_map[FEATURE_NUM]
    df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    min_rows = SEQUENCE_LENGTH + 10
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= min_rows]
    processed = pd.concat([fe_func(g) for g in groups]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=True)

    # Add cross-sectional features
    processed = add_cross_sectional_features(processed)

    # CS-only features
    from utils import CROSS_SECTIONAL_FEATURES
    cs_features = [f'CS_{f}' for f in CROSS_SECTIONAL_FEATURES if f in processed.columns]
    n_feats = len(cs_features)
    print(f"  CS features: {n_feats}")

    # Clean and scale
    processed[cs_features] = processed[cs_features].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=cs_features)
    processed, cs_winsor = winsorize_features(processed, cs_features, 0.01, 0.99)
    scaler = StandardScaler()
    processed[cs_features] = scaler.fit_transform(processed[cs_features])

    del full_df; gc.collect()

    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, cs_features, SEQUENCE_LENGTH)
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del processed, train_tgt, train_rel, train_stk; gc.collect()

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    with open(os.path.join(output_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(cs_winsor, f)
    with open(os.path.join(output_dir, 'cs_features.json'), 'w') as f:
        json.dump(cs_features, f)

    config_data = {
        'feature_dim': n_feats, 'num_stocks': num_stocks,
        'stock_embed_dim': STOCK_EMBED_DIM, 'expert_configs': EXPERT_CONFIGS,
        'cs_only': True,
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
        print(f"Training: {name} (CS-only)")
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

    print(f"\nDone! CS-only")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
