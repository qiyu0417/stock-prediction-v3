"""
GNN v2 (Correlation Graph): replace CrossStockAttention with correlation-graph convolution.
Edges: return correlation > 0.5 → connected. 1-layer GCN with residual + LayerNorm.
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torch.utils.data import DataLoader

from config_stock_emb_8 import *
from train import (create_ranking_dataset_vectorized, RankingDataset, set_seed,
                   calculate_ranking_metrics)
from ensemble_models import StockTransformerExpert, ConvStockExpert
from train_stock_emb_8_loss import (preprocess_with_winsor, collate_fn, _make_criterion,
                                    _train_step, _train_chunked)


# ═══════════════════════════════════════════════════════════
# Graph Stock Convolution — replaces CrossStockAttention
# ═══════════════════════════════════════════════════════════
class GraphStockConv(nn.Module):
    """1-layer GCN with residual + LayerNorm. [B, N, d_model] -> [B, N, d_model]"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def set_adjacency(self, adj_norm):
        """Set pre-computed normalized adjacency matrix [N_full, N_full]."""
        self.register_buffer('adj_norm', adj_norm)

    def forward(self, stock_features):
        # stock_features: [B, N, d_model], N must match adj_norm size
        B, N, D = stock_features.shape
        adj = self.adj_norm[:N, :N].to(stock_features.device)
        support = self.linear(stock_features)
        out = torch.bmm(adj.unsqueeze(0).expand(B, -1, -1), support)
        out = F.relu(out)
        return self.norm(stock_features + self.dropout(out))


def build_correlation_adjacency(stock_ids, full_df, corr_threshold=0.5):
    """Build normalized adjacency from return correlation. Returns [N, N] tensor."""
    N = len(stock_ids)

    # Pivot daily returns
    pvt = full_df.pivot(index='日期', columns='股票代码', values='涨跌幅')
    avail = [s for s in stock_ids if s in pvt.columns]
    pvt = pvt[avail].dropna()

    # Compute correlation for available stocks
    corr = pvt.corr().values  # [M, M]
    M = len(avail)

    # Map available stocks to global indices
    avail_to_global = {s: stock_ids.index(s) for s in avail}

    # Build full NxN adjacency
    adj = np.zeros((N, N), dtype=np.float32)
    for i_local in range(M):
        i_global = avail_to_global[avail[i_local]]
        for j_local in range(M):
            if i_local == j_local:
                continue
            j_global = avail_to_global[avail[j_local]]
            if np.abs(corr[i_local, j_local]) > corr_threshold:
                adj[i_global, j_global] = 1.0

    # Self-loops + normalize
    adj = adj + np.eye(N, dtype=np.float32)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt

    density = (np.abs(adj_norm) > 1e-6).sum() / (N * N)
    avg_deg = adj.sum(1).mean() - 1
    print(f"Corr graph: threshold={corr_threshold}, N={N} ({M} with data), "
          f"density={density:.3f}, avg_deg={avg_deg:.1f}")

    return torch.FloatTensor(adj_norm)


# ═══════════════════════════════════════════════════════════
# Monkey-patch: replace CrossStockAttention with GraphStockConv
# ═══════════════════════════════════════════════════════════
_orig_trans_init = StockTransformerExpert.__init__
_orig_conv_init = ConvStockExpert.__init__

_gnn_adj = None


def _gnn_trans_init(self, input_dim, expert_config, num_stocks):
    _orig_trans_init(self, input_dim, expert_config, num_stocks)
    global _gnn_adj
    gnn = GraphStockConv(self.d_model, expert_config.get('dropout', 0.1))
    if _gnn_adj is not None:
        gnn.set_adjacency(_gnn_adj)
    self.cross_stock_attention = gnn


def _gnn_conv_init(self, input_dim, expert_config, num_stocks):
    _orig_conv_init(self, input_dim, expert_config, num_stocks)
    global _gnn_adj
    d = expert_config.get('hidden_channels', 256)
    gnn = GraphStockConv(d, expert_config.get('dropout', 0.1))
    if _gnn_adj is not None:
        gnn.set_adjacency(_gnn_adj)
    self.cross_stock_attention = gnn


def train_expert_gnn(model, exp_cfg, train_dataset, device, expert_name, loss_type,
                     num_epochs=None):
    """Training loop adapted for GNN: batch_size=1, no chunking (full graph needed)."""
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pin_memory = device.type == 'cuda'
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory)

    criterion = _make_criterion(loss_type)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=LEARNING_RATE * 0.005)

    use_amp = USE_AMP and device.type == 'cuda'
    amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_score = -float('inf')
    best_state = None
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_steps = 0, 0
        metrics_sum = {}

        for batch in loader:
            seq = batch['sequences'].to(device, non_blocking=pin_memory)
            rel = batch['relevance'].to(device, non_blocking=pin_memory)
            tgt = batch['targets'].to(device, non_blocking=pin_memory)
            masks = batch['masks'].to(device, non_blocking=pin_memory)

            loss_val, m = _train_step(
                model, seq, rel, tgt, masks, criterion, optimizer, amp_scaler, use_amp)
            if loss_val is not None:
                total_loss += loss_val
                n_steps += 1
                for k, v in m.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v

        scheduler.step()

        if n_steps > 0:
            total_loss /= n_steps
            for k in metrics_sum:
                metrics_sum[k] /= n_steps

        score = metrics_sum.get('final_score', 0)

        if (epoch + 1) % 5 == 0:
            print(f"  [{expert_name}] Epoch {epoch + 1:2d}/{num_epochs} | "
                  f"Loss: {total_loss:.4f} | Score: {score:.4f} | "
                  f"Patience: {patience}/{EARLY_STOPPING_PATIENCE}")

        if score > best_score + EARLY_STOPPING_MIN_DELTA:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"  [{expert_name}] Early stop! Best: {best_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        return best_score
    return best_score


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss
    num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"GNN Training | Loss: {loss_type} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_gnn_corr')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # ── Build correlation adjacency ──
    global _gnn_adj
    _gnn_adj = build_correlation_adjacency(stock_ids, full_df, corr_threshold=0.5)

    # ── Monkey-patch ──
    StockTransformerExpert.__init__ = _gnn_trans_init
    ConvStockExpert.__init__ = _gnn_conv_init

    # ── Override conv_deep for laptop GPU ──
    expert_configs = []
    for ec in EXPERT_CONFIGS:
        ec = dict(ec)
        if ec['name'] == 'conv_deep':
            ec['hidden_channels'] = 256
        expert_configs.append(ec)

    # ── Preprocess ──
    processed, features, scaler, winsor_bounds = preprocess_with_winsor(full_df, stockid2idx)
    n_feats = len(features)
    del full_df
    gc.collect()

    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, features, SEQUENCE_LENGTH)
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del processed, train_tgt, train_rel, train_stk
    gc.collect()

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    with open(os.path.join(output_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f)

    config_data = {
        'feature_dim': n_feats, 'num_stocks': num_stocks,
        'stock_embed_dim': STOCK_EMBED_DIM, 'expert_configs': expert_configs,
        'gnn': True, 'graph_type': 'return_correlation_th0.5',
    }
    with open(os.path.join(output_dir, 'ensemble_config.json'), 'w') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    # ── Train experts ──
    all_results = {}
    for exp_cfg in expert_configs:
        name = exp_cfg['name']
        model_path = os.path.join(output_dir, f'expert_{name}.pth')
        if os.path.exists(model_path):
            print(f"\n  SKIP {name}: already trained")
            all_results[name] = 0.0
            continue

        print(f"\n{'=' * 50}")
        print(f"Training: {name} (GNN)")
        model = StockTransformerExpert(n_feats, exp_cfg, num_stocks) \
            if exp_cfg['type'] == 'transformer' \
            else ConvStockExpert(n_feats, exp_cfg, num_stocks)
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
        model.to(device)

        score = train_expert_gnn(model, exp_cfg, dataset, device, name, loss_type, num_epochs)
        all_results[name] = score
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: expert_{name}.pth (score={score:.4f})")
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone! GNN")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")

    # Restore original inits
    StockTransformerExpert.__init__ = _orig_trans_init
    ConvStockExpert.__init__ = _orig_conv_init


if __name__ == '__main__':
    main()
