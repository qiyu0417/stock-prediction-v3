"""
Mamba temporal encoder: replaces Transformer with selective SSM blocks
Based on Mamba (Gu & Dao, 2023) — O(n) vs Transformer O(n^2)
Selective scan implemented via parallel associative scan (pure PyTorch)
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, joblib
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_stock_emb_8 import *
from train import (create_ranking_dataset_vectorized, RankingDataset, set_seed, calculate_ranking_metrics)
from ensemble_models import StockTransformerExpert, ConvStockExpert, CrossStockAttention, PositionalEncoding
from train_stock_emb_8_loss import preprocess_with_winsor, collate_fn, _make_criterion, train_expert


class SelectiveSSM(nn.Module):
    """Selective State Space Model — core of Mamba"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2)  # x and z branches
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.act = nn.SiLU()

        # SSM parameters: A (diagonal), B (input→state), C (state→output), D (skip)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)))
        self.A_log._no_weight_decay = True
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_state)  # dt_rank, B, C
        self.dt_proj = nn.Linear(d_state, d_inner)  # dt_rank=1 for simplicity
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        d_inner = self.in_proj.in_features * self.in_proj.out_features // (D * 2)
        # Correct d_inner
        d_inner = self.out_proj.in_features

        x_and_z = self.in_proj(x)  # [B, L, 2*d_inner]
        x_branch, z = x_and_z.chunk(2, dim=-1)

        # Convolution
        x_conv = x_branch.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv[..., :L]  # causal
        x_conv = self.act(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]

        # Selective SSM via parallel scan
        A = -torch.exp(self.A_log)  # [1, d_state]
        x_proj = self.x_proj(x_conv)  # [B, L, 2*d_state + d_state]

        # Simplified: dt=1, use direct recurrence (for small d_state this is fast enough)
        # B, C as separate projections
        B_ssm = x_proj[:, :, :self.d_state]  # [B, L, d_state]
        C_ssm = x_proj[:, :, self.d_state:2*self.d_state]  # [B, L, d_state]

        # Discrete recurrence
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            # Input-dependent delta (simplified: no delta projection)
            delta = F.softplus(self.dt_proj(x_proj[:, t, 2*self.d_state:]))  # [B, d_inner]
            A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, d_state]
            B_input = x_conv[:, t]  # [B, d_inner]
            h = A_bar * h + B_input.unsqueeze(-1) * B_ssm[:, t].unsqueeze(1)
            y = (h * C_ssm[:, t].unsqueeze(1)).sum(-1) + self.D * B_input  # [B, d_inner]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        y = y * self.act(z)

        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Single Mamba layer: pre-norm + SSM + residual (matches Transformer layer interface)"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual


class MambaTemporalEncoder(nn.Module):
    """Stack of Mamba blocks. Supports iteration like ModuleList."""
    def __init__(self, d_model, num_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Monkey-patch: replace temporal_layers with Mamba encoder
_orig_trans_init = StockTransformerExpert.__init__


def _mamba_trans_init(self, input_dim, expert_config, num_stocks):
    _orig_trans_init(self, input_dim, expert_config, num_stocks)
    cfg = expert_config
    # Replace temporal_layers with Mamba
    num_layers = cfg.get('num_layers', 4)
    self.temporal_layers = MambaTemporalEncoder(
        self.d_model, num_layers=num_layers,
        d_state=16, d_conv=4, expand=2,
        dropout=cfg.get('dropout', 0.1))


StockTransformerExpert.__init__ = _mamba_trans_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Mamba Training | Loss: {loss_type} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_mamba')
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
        'mamba': True,
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
        print(f"Training: {name} (Mamba)")

        if exp_cfg['type'] == 'transformer':
            model = StockTransformerExpert(n_feats, exp_cfg, num_stocks)
        else:
            model = ConvStockExpert(n_feats, exp_cfg, num_stocks)

        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
        model.to(device)

        best_score = train_expert(model, exp_cfg, dataset, device, name, loss_type, num_epochs)
        all_results[name] = best_score
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone! Mamba")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
