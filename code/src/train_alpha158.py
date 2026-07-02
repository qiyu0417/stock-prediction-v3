"""
Alpha158 momentum expert: only price/volume momentum factors, no technical indicators
Different signal perspective from full 197-dim models
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, torch, joblib
from torch.utils.data import DataLoader

from config_stock_emb_8 import *
from train import (create_ranking_dataset_vectorized, RankingDataset, set_seed,
                   calculate_ranking_metrics, _ALPHA_158_COLS)
from ensemble_models import StockTransformerExpert, ConvStockExpert
from train_stock_emb_8_loss import preprocess_with_winsor, collate_fn, _make_criterion, train_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Alpha158 Training | Loss: {loss_type} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_alpha158')
    os.makedirs(output_dir, exist_ok=True)

    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # Full preprocessing (all 197 features for winsor/scaler consistency)
    processed, features_all, scaler, winsor_bounds = preprocess_with_winsor(full_df, stockid2idx)
    del full_df; gc.collect()

    # Filter to Alpha158 only
    alpha_features = [f for f in _ALPHA_158_COLS if f in features_all]
    n_feats = len(alpha_features)
    print(f"  Alpha158 features: {n_feats} / original {len(features_all)}")

    # Re-filter winsor bounds and scaler to alpha features only
    alpha_winsor = {k: v for k, v in winsor_bounds.items() if k in alpha_features}

    # Build dataset with only alpha features
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, alpha_features, SEQUENCE_LENGTH)
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del processed, train_tgt, train_rel, train_stk; gc.collect()

    # Save scaler (full) and winsor (alpha-filtered)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    with open(os.path.join(output_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f)  # save full winsor for eval preprocessing
    with open(os.path.join(output_dir, 'alpha_features.json'), 'w') as f:
        json.dump(alpha_features, f)

    config_data = {
        'feature_dim': n_feats, 'num_stocks': num_stocks,
        'stock_embed_dim': STOCK_EMBED_DIM, 'expert_configs': EXPERT_CONFIGS,
        'alpha158': True,
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
        print(f"Training: {name} (Alpha158)")
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

    print(f"\nDone! Alpha158")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
