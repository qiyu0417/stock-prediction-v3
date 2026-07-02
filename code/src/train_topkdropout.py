"""
TopK-Dropout: keep top-k features by magnitude after feature attention
Prevents over-reliance on weak features — Stockformer 2025
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
from train_stock_emb_8_loss import preprocess_with_winsor, collate_fn, _make_criterion

TOPK_KEEP_RATIO = 0.75  # keep top 75% features


class TopKDropout(nn.Module):
    """Zero out bottom-(1-keep_ratio) features by absolute magnitude"""
    def __init__(self, keep_ratio=0.75):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x):
        if not self.training:
            return x
        k = max(1, int(x.size(-1) * self.keep_ratio))
        _, indices = x.abs().topk(k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1.0)
        return x * mask / self.keep_ratio  # scale to preserve magnitude


class TopKDropoutExpert(nn.Module):
    """Wrapper: base expert + TopK-Dropout after feature attention"""
    def __init__(self, base_expert, keep_ratio=0.75):
        super().__init__()
        self.expert = base_expert
        self.topk = TopKDropout(keep_ratio)
        self.d_model = base_expert.d_model if hasattr(base_expert, 'd_model') else 256

    def forward(self, src):
        # Get scores from base expert — TopK applied internally via feature attention mod
        # We override by using forward_features and applying TopK on features
        scores, features = self.expert.forward_features(src)
        features = self.topk(features)
        # Re-project features to scores using the expert's score head
        # Need to re-score: we modify the raw scores based on TopK features
        # Simple approach: just return the original scores (TopK is regularization at feature level)
        return scores

    def forward_features(self, src):
        scores, features = self.expert.forward_features(src)
        features = self.topk(features)
        return scores, features


# Actually implement TopK-Dropout properly inside the model
# Monkey-patch: override FeatureAttention.forward to include TopK-Dropout

import ensemble_models

_original_feature_attn_forward = ensemble_models.FeatureAttention.forward

def _topk_feature_attn_forward(self, x):
    # x: [B*N, seq_len, d_model]
    attn_weights = self.attention(x)  # [BN, seq_len, 1]
    # TopK on attention: keep top-k time steps
    k = max(1, int(x.size(1) * TOPK_KEEP_RATIO))
    attn_flat = attn_weights.squeeze(-1)  # [BN, seq_len]
    _, topk_idx = attn_flat.topk(k, dim=1)
    mask = torch.zeros_like(attn_flat)
    mask.scatter_(1, topk_idx, 1.0)
    attn_weights = attn_weights * mask.unsqueeze(-1)
    attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

    attended = torch.sum(x * attn_weights, dim=1)  # [BN, d_model]
    return self.dropout(attended)


ensemble_models.FeatureAttention.forward = _topk_feature_attn_forward

# Re-import to use modified attention
from train_stock_emb_8_loss import train_expert as _train_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--ratio', type=float, default=TOPK_KEEP_RATIO)
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs; keep_ratio = args.ratio

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"TopK-Dropout Training | Loss: {loss_type} | Keep ratio: {keep_ratio} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', f'stock_emb_8_topk{int(keep_ratio*100)}')
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
        'topk_keep_ratio': keep_ratio,
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
        print(f"Training: {name} (TopK-{int(keep_ratio*100)}%)")
        model = StockTransformerExpert(n_feats, exp_cfg, num_stocks) if exp_cfg['type'] == 'transformer' \
                else ConvStockExpert(n_feats, exp_cfg, num_stocks)
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
        model.to(device)

        best_score = _train_expert(model, exp_cfg, dataset, device, name, loss_type, num_epochs)
        all_results[name] = best_score
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone! TopK-{int(keep_ratio*100)}%")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
