"""
GLU Ranking Head: replace Linear+ReLU in ranking_layers with GatedLinear
Extends the GLU idea from FeatureAttention to the scoring MLP
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_stock_emb_8 import *
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   create_ranking_dataset_vectorized, RankingDataset, set_seed,
                   calculate_ranking_metrics)
from ensemble_models import StockTransformerExpert, ConvStockExpert
from train_stock_emb_8_loss import preprocess_with_winsor, collate_fn, _make_criterion, train_expert


class GatedLinear(nn.Module):
    """Linear layer with sigmoid gating"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.value = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        v = self.value(x)
        g = torch.sigmoid(self.gate(x))
        return self.dropout(self.norm(v * g))


# Monkey-patch ranking_layers construction in both experts
_orig_stock_transformer_init = StockTransformerExpert.__init__
_orig_conv_stock_init = ConvStockExpert.__init__


def _patched_transformer_init(self, input_dim, expert_config, num_stocks):
    _orig_stock_transformer_init(self, input_dim, expert_config, num_stocks)
    cfg = expert_config
    d_model = self.d_model
    dropout = cfg.get('dropout', 0.1)
    # Replace ranking_layers with GLU version
    self.ranking_layers = nn.Sequential(
        GatedLinear(d_model, d_model, dropout),
        GatedLinear(d_model, d_model // 2, dropout),
        nn.LayerNorm(d_model // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


def _patched_conv_init(self, input_dim, expert_config, num_stocks):
    _orig_conv_stock_init(self, input_dim, expert_config, num_stocks)
    cfg = expert_config
    hidden = self.d_model
    dropout = cfg.get('dropout', 0.1)
    self.ranking_layers = nn.Sequential(
        GatedLinear(hidden, hidden, dropout),
        GatedLinear(hidden, hidden // 2, dropout),
        nn.LayerNorm(hidden // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


StockTransformerExpert.__init__ = _patched_transformer_init
ConvStockExpert.__init__ = _patched_conv_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"GLU Rank Training | Loss: {loss_type} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_glurank')
    os.makedirs(output_dir, exist_ok=True)

    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    processed, features, scaler, winsor_bounds = preprocess_with_winsor(full_df, stockid2idx)
    n_feats = len(features)
    del full_df; gc.collect()

    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, features, SEQUENCE_LENGTH)
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del processed, train_tgt, train_rel, train_stk; gc.collect()

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    with open(os.path.join(output_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f)

    config_data = {
        'feature_dim': n_feats, 'num_stocks': num_stocks,
        'stock_embed_dim': STOCK_EMBED_DIM, 'expert_configs': EXPERT_CONFIGS,
        'glurank': True,
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
        print(f"Training: {name} (GLU Rank)")
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

    print(f"\nDone! GLU Rank")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
