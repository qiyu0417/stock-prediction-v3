"""
Multi-task training: ranking loss + direction (up/down) prediction
Model returns (scores, features) — direction head added on features
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

DIRECTION_WEIGHT = 0.3  # weight of direction loss vs ranking loss


class DirectionHead(nn.Module):
    """Simple 2-layer MLP for binary direction prediction"""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)  # [B*N]


def train_expert_multitask(model, direction_head, exp_cfg, train_dataset, device, expert_name, loss_type, num_epochs=None):
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pin_memory = device.type == 'cuda'
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory)

    criterion = _make_criterion(loss_type)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(direction_head.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
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
        direction_head.train()
        total_loss, n_steps = 0, 0
        metrics_sum = {}

        for batch in loader:
            seq = batch['sequences'].to(device, non_blocking=pin_memory)
            rel = batch['relevance'].to(device, non_blocking=pin_memory)
            tgt = batch['targets'].to(device, non_blocking=pin_memory)
            masks = batch['masks'].to(device, non_blocking=pin_memory)
            n_stocks = seq.size(1)

            if n_stocks <= chunk_size:
                scores, features = model.forward_features(seq)
                # Direction labels: sign of tgt (up=1, down=0)
                dir_labels = (tgt > 0).float().view(-1)
                dir_logits = direction_head(features)  # [B*N]

                masked = scores * masks + (1 - masks) * (-1e9)
                loss = None
                for i in range(seq.size(0)):
                    valid_idx = masks[i].nonzero().squeeze()
                    if valid_idx.numel() <= 1:
                        continue
                    if valid_idx.dim() == 0:
                        valid_idx = valid_idx.unsqueeze(0)
                    valid_pred = masked[i][valid_idx]
                    valid_rel = rel[i][valid_idx].float()
                    li = criterion(valid_pred.unsqueeze(0), valid_rel.unsqueeze(0))
                    loss = loss + li if loss is not None else li

                if loss is not None:
                    # Direction loss on valid stocks only
                    valid_mask_flat = masks.view(-1).bool()
                    if valid_mask_flat.sum() > 0:
                        dir_l = bce_loss(dir_logits[valid_mask_flat], dir_labels[valid_mask_flat])
                        loss = loss + DIRECTION_WEIGHT * dir_l

                    optimizer.zero_grad()
                    if use_amp:
                        amp_scaler.scale(loss).backward()
                        amp_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        torch.nn.utils.clip_grad_norm_(direction_head.parameters(), MAX_GRAD_NORM)
                        amp_scaler.step(optimizer)
                        amp_scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        torch.nn.utils.clip_grad_norm_(direction_head.parameters(), MAX_GRAD_NORM)
                        optimizer.step()

                    total_loss += loss.item()
                    n_steps += 1
                    with torch.no_grad():
                        mc = calculate_ranking_metrics(masked.detach(), tgt * masks, masks, k=5)
                    for k, v in mc.items():
                        metrics_sum[k] = metrics_sum.get(k, 0) + v
            else:
                # Chunked training for large n_stocks
                optimizer.zero_grad()
                n_chunks = (n_stocks + chunk_size - 1) // chunk_size
                for start in range(0, n_stocks, chunk_size):
                    end = min(start + chunk_size, n_stocks)
                    seq_c = seq[:, start:end].contiguous()
                    rel_c = rel[:, start:end].contiguous()
                    tgt_c = tgt[:, start:end].contiguous()
                    mask_c = masks[:, start:end].contiguous()

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        scores_c, features_c = model.forward_features(seq_c)
                        dir_labels_c = (tgt_c > 0).float().view(-1)
                        dir_logits_c = direction_head(features_c)

                        masked_c = scores_c * mask_c + (1 - mask_c) * (-1e9)
                        loss_c = None
                        for i in range(seq_c.size(0)):
                            valid_idx = mask_c[i].nonzero().squeeze()
                            if valid_idx.numel() <= 1:
                                continue
                            if valid_idx.dim() == 0:
                                valid_idx = valid_idx.unsqueeze(0)
                            valid_pred = masked_c[i][valid_idx]
                            valid_rel_c = rel_c[i][valid_idx].float()
                            li = criterion(valid_pred.unsqueeze(0), valid_rel_c.unsqueeze(0))
                            loss_c = loss_c + li if loss_c is not None else li

                        if loss_c is not None:
                            valid_mask_flat_c = mask_c.view(-1).bool()
                            if valid_mask_flat_c.sum() > 0:
                                dir_l = bce_loss(dir_logits_c[valid_mask_flat_c], dir_labels_c[valid_mask_flat_c])
                                loss_c = loss_c + DIRECTION_WEIGHT * dir_l
                            loss_c = loss_c / n_chunks

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
                torch.nn.utils.clip_grad_norm_(direction_head.parameters(), MAX_GRAD_NORM)
                if use_amp:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                n_steps += 1

        scheduler.step()

        # Average metrics
        if n_steps > 0:
            for k in metrics_sum:
                metrics_sum[k] /= n_steps

        score = metrics_sum.get('final_score', 0)
        if (epoch + 1) % 5 == 0:
            print(f"  [{expert_name}] Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Loss: {total_loss/max(n_steps,1):.4f} | Score: {score:.4f} | "
                  f"耐心: {patience}/{EARLY_STOPPING_PATIENCE}")

        if score > best_score + EARLY_STOPPING_MIN_DELTA:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_dir_state = {k: v.cpu().clone() for k, v in direction_head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"  [{expert_name}] 早停! 最佳: {best_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        direction_head.load_state_dict(best_dir_state)
    return best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Multi-Task Training | Loss: {loss_type} | Direction weight: {DIRECTION_WEIGHT} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_multitask')
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
        'multitask': True,
    }
    with open(os.path.join(output_dir, 'ensemble_config.json'), 'w') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    all_results = {}
    for exp_cfg in EXPERT_CONFIGS:
        name = exp_cfg['name']
        model_path = os.path.join(output_dir, f'expert_{name}.pth')
        dir_path = os.path.join(output_dir, f'direction_{name}.pth')
        if os.path.exists(model_path):
            print(f"\n  SKIP {name}: already trained")
            all_results[name] = 0.0
            continue

        print(f"\n{'='*50}")
        print(f"Training: {name} (multitask)")

        model = StockTransformerExpert(n_feats, exp_cfg, num_stocks) if exp_cfg['type'] == 'transformer' \
                else ConvStockExpert(n_feats, exp_cfg, num_stocks)
        d_model = exp_cfg.get('d_model', exp_cfg.get('hidden_channels', 256))
        direction_head = DirectionHead(d_model)
        print(f"  params: {sum(p.numel() for p in model.parameters()):,} + "
              f"{sum(p.numel() for p in direction_head.parameters()):,}")

        model.to(device); direction_head.to(device)
        best_score = train_expert_multitask(model, direction_head, exp_cfg, dataset, device, name, loss_type, num_epochs)
        all_results[name] = best_score

        torch.save(model.state_dict(), model_path)
        torch.save(direction_head.state_dict(), dir_path)
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model, direction_head; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone! Multi-task")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
