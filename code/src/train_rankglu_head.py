"""
RankGLU score head: residual bottleneck GLU + IC-augmented loss
Based on RankGLU paper (Xiao et al., 2026): bounded residual score formation
Key insight: GLU in score HEAD (not attention), with IC loss term
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, joblib
from torch.utils.data import DataLoader

from config_stock_emb_8 import *
from train import (create_ranking_dataset_vectorized, RankingDataset, set_seed, calculate_ranking_metrics)
from ensemble_models import StockTransformerExpert, ConvStockExpert
from train_stock_emb_8_loss import preprocess_with_winsor, collate_fn

IC_LOSS_WEIGHT = 0.1
BOTTLENECK = 128
GAMMA = 0.1  # residual GLU scaling


class ResidualBottleneckGLU(nn.Module):
    """
    RankGLU score head: score = Linear(x) + gamma * BottleneckGLU(x)
    Direct linear path + bounded multiplicative branch
    """
    def __init__(self, d_model, d_score=1, bottleneck=128, dropout=0.05):
        super().__init__()
        self.gamma = GAMMA
        self.norm = nn.LayerNorm(d_model)
        # Direct linear path
        self.linear_direct = nn.Linear(d_model, d_score)
        # Gated bottleneck branch
        self.value_proj = nn.Linear(d_model, bottleneck)
        self.gate_proj = nn.Linear(d_model, bottleneck)
        self.out_proj = nn.Linear(bottleneck, d_score)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        xn = self.norm(x)
        # Direct path
        score = self.linear_direct(xn)
        # Gated bottleneck path
        v = self.value_proj(xn)
        g = torch.sigmoid(self.gate_proj(xn))
        gated = self.out_proj(self.dropout(v * g))
        return score + self.gamma * gated


def pearson_ic(pred, target):
    """Compute Pearson IC (cross-sectional rank correlation)"""
    pred = pred.float()
    target = target.float()
    pred_mean = pred.mean(dim=-1, keepdim=True)
    target_mean = target.mean(dim=-1, keepdim=True)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    numerator = (pred_centered * target_centered).sum(dim=-1)
    denominator = torch.sqrt((pred_centered ** 2).sum(dim=-1) * (target_centered ** 2).sum(dim=-1)) + 1e-8
    return (numerator / denominator).mean()


# Monkey-patch score_head in both experts
_orig_trans_init = StockTransformerExpert.__init__
_orig_conv_init = ConvStockExpert.__init__


def _rglu_trans_init(self, input_dim, expert_config, num_stocks):
    _orig_trans_init(self, input_dim, expert_config, num_stocks)
    self.score_head = ResidualBottleneckGLU(self.d_model // 2, bottleneck=BOTTLENECK)


def _rglu_conv_init(self, input_dim, expert_config, num_stocks):
    _orig_conv_init(self, input_dim, expert_config, num_stocks)
    self.score_head = ResidualBottleneckGLU(self.d_model // 2, bottleneck=BOTTLENECK)


StockTransformerExpert.__init__ = _rglu_trans_init
ConvStockExpert.__init__ = _rglu_conv_init


def train_expert_rankglu(model, exp_cfg, train_dataset, device, expert_name, num_epochs=None):
    """Training with IC-augmented hybrid loss"""
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pin_memory = device.type == 'cuda'
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory)

    from train_stock_emb_8_loss import _make_criterion
    criterion = _make_criterion('hybrid')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=LEARNING_RATE * 0.005)

    use_amp = USE_AMP and device.type == 'cuda'
    amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999

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
            n_stocks = seq.size(1)

            if n_stocks <= chunk_size:
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    scores = model(seq)
                    masked = scores * masks + (1 - masks) * (-1e9)
                    loss = None
                    ic_loss = 0.0
                    n_valid_batches = 0
                    B = seq.size(0)
                    for i in range(B):
                        valid_idx = masks[i].nonzero().squeeze()
                        if valid_idx.numel() <= 1:
                            continue
                        if valid_idx.dim() == 0:
                            valid_idx = valid_idx.unsqueeze(0)
                        li = criterion(masked[i][valid_idx].unsqueeze(0), rel[i][valid_idx].float().unsqueeze(0))
                        loss = loss + li if loss is not None else li
                        # IC loss on valid predictions
                        ic_loss += pearson_ic(scores[i][valid_idx], tgt[i][valid_idx])
                        n_valid_batches += 1
                    if n_valid_batches > 0:
                        ic_loss /= n_valid_batches
                if loss is not None:
                    loss = loss - IC_LOSS_WEIGHT * ic_loss  # maximize IC = minimize -IC
                    if use_amp:
                        amp_scaler.scale(loss).backward()
                        amp_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        amp_scaler.step(optimizer)
                        amp_scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                    total_loss += loss.item()
                    n_steps += 1
                    with torch.no_grad():
                        mc = calculate_ranking_metrics(masked.detach(), tgt * masks, masks, k=5)
                    for k, v in mc.items():
                        metrics_sum[k] = metrics_sum.get(k, 0) + v
            else:
                optimizer.zero_grad()
                n_chunks = (n_stocks + chunk_size - 1) // chunk_size
                for start in range(0, n_stocks, chunk_size):
                    end = min(start + chunk_size, n_stocks)
                    seq_c = seq[:, start:end].contiguous()
                    rel_c = rel[:, start:end].contiguous()
                    tgt_c = tgt[:, start:end].contiguous()
                    mask_c = masks[:, start:end].contiguous()
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        scores_c = model(seq_c)
                        masked_c = scores_c * mask_c + (1 - mask_c) * (-1e9)
                        loss_c = None
                        ic_loss_c = 0.0
                        nv = 0
                        for i in range(seq_c.size(0)):
                            valid_idx = mask_c[i].nonzero().squeeze()
                            if valid_idx.numel() <= 1:
                                continue
                            if valid_idx.dim() == 0:
                                valid_idx = valid_idx.unsqueeze(0)
                            li = criterion(masked_c[i][valid_idx].unsqueeze(0), rel_c[i][valid_idx].float().unsqueeze(0))
                            loss_c = loss_c + li if loss_c is not None else li
                            ic_loss_c += pearson_ic(scores_c[i][valid_idx], tgt_c[i][valid_idx])
                            nv += 1
                        if nv > 0:
                            ic_loss_c /= nv
                    if loss_c is not None:
                        loss_c = (loss_c - IC_LOSS_WEIGHT * ic_loss_c) / n_chunks
                        if use_amp:
                            amp_scaler.scale(loss_c).backward()
                        else:
                            loss_c.backward()
                        total_loss += loss_c.item()
                        with torch.no_grad():
                            mc = calculate_ranking_metrics(masked_c.detach(), tgt_c * mask_c, mask_c, k=5)
                        for k, v in mc.items():
                            metrics_sum[k] = metrics_sum.get(k, 0) + v / n_chunks
                if use_amp:
                    amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if use_amp:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                n_steps += 1

        scheduler.step()

        if n_steps > 0:
            total_loss /= n_steps
            for k in metrics_sum:
                metrics_sum[k] /= n_steps

        score = metrics_sum.get('final_score', 0)
        if (epoch + 1) % 5 == 0:
            print(f"  [{expert_name}] Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Loss: {total_loss:.4f} | Score: {score:.4f} | "
                  f"耐心: {patience}/{EARLY_STOPPING_PATIENCE}")

        if score > best_score + EARLY_STOPPING_MIN_DELTA:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"  [{expert_name}] 早停! 最佳: {best_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"RankGLU Head Training | Bottleneck: {BOTTLENECK} | IC weight: {IC_LOSS_WEIGHT} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_rankglu_head')
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
        'rankglu_head': True, 'bottleneck': BOTTLENECK, 'ic_weight': IC_LOSS_WEIGHT,
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
        print(f"Training: {name} (RankGLU Head)")
        model = StockTransformerExpert(n_feats, exp_cfg, num_stocks) if exp_cfg['type'] == 'transformer' \
                else ConvStockExpert(n_feats, exp_cfg, num_stocks)
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
        model.to(device)

        best_score = train_expert_rankglu(model, exp_cfg, dataset, device, name, num_epochs)
        all_results[name] = best_score
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone! RankGLU Head")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
