"""
训练: 158+39+CS 截面特征 + Hybrid loss
feature_dim = 197 + 25 CS = 222
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkey-patch: override FEATURE_NUM before importing train_stock_emb_8_loss
import config_stock_emb_8
config_stock_emb_8.FEATURE_NUM = '158+39+CS'

import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

from config_stock_emb_8 import *
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   create_ranking_dataset_vectorized, RankingDataset, set_seed)
from utils import add_cross_sectional_features, engineer_features_158plus39

# Register CS feature engineer
def _engineer_158plus39_cs(df):
    df_out = engineer_features_158plus39(df, add_market=False)
    df_out = add_cross_sectional_features(df_out)
    return df_out

feature_engineer_func_map['158+39+CS'] = _engineer_158plus39_cs

from train_stock_emb_8_loss import (preprocess_with_winsor, train_expert, _make_criterion,
                                     collate_fn, calculate_ranking_metrics)
from ensemble_models import StockTransformerExpert, ConvStockExpert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss
    num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"CS Features Training | Loss: {loss_type} | Feature: {FEATURE_NUM} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', f'stock_emb_8_{loss_type}_cs')
    os.makedirs(output_dir, exist_ok=True)

    print("\n加载数据...")
    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    print(f"  总数据: {len(full_df)} 行")

    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    print("\n预处理...")
    processed, features, scaler, winsor_bounds = preprocess_with_winsor(full_df, stockid2idx)
    n_feats = len(features)
    print(f"  特征维度: {n_feats}")

    del full_df; gc.collect()

    print("\n构建排名数据集...")
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, features, SEQUENCE_LENGTH)
    print(f"  训练天数: {len(train_seq)}")
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del processed, train_tgt, train_rel, train_stk; gc.collect()

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    with open(os.path.join(output_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f)
    print(f"Saved scaler + winsor_bounds → {output_dir}/")

    # Save config
    config_data = {
        'feature_dim': n_feats,
        'num_stocks': num_stocks,
        'stock_embed_dim': STOCK_EMBED_DIM,
        'expert_configs': EXPERT_CONFIGS,
        'feature_num': FEATURE_NUM,
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
        print(f"Training: {name} (type={exp_cfg['type']})")
        model = StockTransformerExpert(n_feats, exp_cfg, num_stocks) if exp_cfg['type'] == 'transformer' \
                else ConvStockExpert(n_feats, exp_cfg, num_stocks)
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
        model.to(device)

        best_score = train_expert(model, exp_cfg, dataset, device, name, loss_type, num_epochs)
        all_results[name] = best_score

        torch.save(model.state_dict(), os.path.join(output_dir, f'expert_{name}.pth'))
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone! {FEATURE_NUM} features")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
