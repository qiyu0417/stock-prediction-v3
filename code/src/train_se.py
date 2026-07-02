"""
Squeeze-Excitation: channel attention after feature aggregation
Proven in CV (SENet), lightweight: adds negligible params
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


class SEBlock(nn.Module):
    """Channel-wise Squeeze-Excitation"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


# Monkey-patch: insert SEBlock after FeatureAttention in both experts
_orig_trans_init = StockTransformerExpert.__init__
_orig_conv_init = ConvStockExpert.__init__


def _se_trans_init(self, input_dim, expert_config, num_stocks):
    _orig_trans_init(self, input_dim, expert_config, num_stocks)
    self.se_block = SEBlock(self.d_model)


def _se_conv_init(self, input_dim, expert_config, num_stocks):
    _orig_conv_init(self, input_dim, expert_config, num_stocks)
    self.se_block = SEBlock(self.d_model)


StockTransformerExpert.__init__ = _se_trans_init
ConvStockExpert.__init__ = _se_conv_init

# Also monkey-patch forward to apply SE after FeatureAttention
_orig_trans_forward = StockTransformerExpert.forward
_orig_conv_forward = ConvStockExpert.forward


def _se_trans_forward(self, src):
    batch_size, num_stocks, seq_len, feature_dim = src.shape
    src_flat = src.view(batch_size * num_stocks, seq_len, feature_dim)

    if self.industry_embed_dim > 0:
        tech = src_flat[..., :-14]
        ind = src_flat[..., -14:]
        x = self.input_proj(tech)
        i = self.industry_compressor(ind)
        x = torch.cat([x, i], dim=-1)
        x = self.industry_fusion(x)
    else:
        x = self.input_proj(src_flat)
    x = self.pos_encoder(x)

    for layer in self.temporal_layers:
        x = layer(x)

    x = self.feature_attention(x)  # [B*N, d_model]
    x = self.se_block(x)  # SE channel attention

    stock_features = x.view(batch_size, num_stocks, -1)
    if self.stock_embed_dim > 0:
        stock_ids = torch.arange(num_stocks, device=src.device)
        stock_emb = self.stock_embedding(stock_ids)
        stock_emb = stock_emb.unsqueeze(0).expand(batch_size, -1, -1)
        stock_features = stock_features + self.stock_emb_proj(stock_emb)
    stock_features = self.cross_stock_attention(stock_features)
    stock_features = stock_features.view(batch_size * num_stocks, -1)

    ranking_features = self.ranking_layers(stock_features)
    scores = self.score_head(ranking_features)
    return scores.view(batch_size, num_stocks)


def _se_conv_forward(self, src):
    batch_size, num_stocks, seq_len, feature_dim = src.shape
    x = src.view(batch_size * num_stocks, seq_len, feature_dim)

    if self.industry_embed_dim > 0:
        tech = x[..., :-14]
        ind = x[..., -14:]
        x = self.input_proj(tech)
        i = self.industry_compressor(ind)
        x = torch.cat([x, i], dim=-1)
        x = self.industry_fusion(x)
    else:
        x = self.input_proj(x)
    x = self.input_norm(x)
    x = self.input_dropout(x)

    for block in self.tcn_blocks:
        x = block(x)

    x = self.feature_attention(x)
    x = self.se_block(x)  # SE channel attention

    stock_features = x.view(batch_size, num_stocks, -1)
    if self.stock_embed_dim > 0:
        stock_ids = torch.arange(num_stocks, device=src.device)
        stock_emb = self.stock_embedding(stock_ids)
        stock_emb = stock_emb.unsqueeze(0).expand(batch_size, -1, -1)
        stock_features = stock_features + self.stock_emb_proj(stock_emb)
    stock_features = self.cross_stock_attention(stock_features)
    stock_features = stock_features.view(batch_size * num_stocks, -1)

    ranking_features = self.ranking_layers(stock_features)
    scores = self.score_head(ranking_features)
    return scores.view(batch_size, num_stocks)


StockTransformerExpert.forward = _se_trans_forward
ConvStockExpert.forward = _se_conv_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"SE Training | Loss: {loss_type} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_se')
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
        'se_block': True,
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
        print(f"Training: {name} (SE)")
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

    print(f"\nDone! SE")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
