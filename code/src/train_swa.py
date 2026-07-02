"""
SWA (Stochastic Weight Averaging): saves CPU checkpoints, averages after training.
Uses proven training functions from train_stock_emb_8_loss to avoid OOM.
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from torch.utils.data import DataLoader

from config_stock_emb_8 import *
from train import (create_ranking_dataset_vectorized, RankingDataset, set_seed, calculate_ranking_metrics)
from ensemble_models import StockTransformerExpert, ConvStockExpert
from train_stock_emb_8_loss import (preprocess_with_winsor, collate_fn, _make_criterion,
                                     _train_step, _train_chunked)

SWA_START = 0.6  # start saving checkpoints at 60% of training


def _average_state_dicts(state_dicts):
    """Average a list of CPU state_dicts, returns averaged state_dict."""
    avg = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg[key] = stacked.mean(dim=0)
    return avg


def train_expert_swa(model, exp_cfg, train_dataset, device, expert_name, loss_type, num_epochs=None):
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pin_memory = device.type == 'cuda'
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory)

    criterion = _make_criterion(loss_type)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=LEARNING_RATE * 0.005)

    swa_start_epoch = int(num_epochs * SWA_START)
    swa_checkpoints = []

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
                loss_val, m = _train_step(
                    model, seq, rel, tgt, masks, criterion, optimizer, amp_scaler, use_amp)
                if loss_val is not None:
                    total_loss += loss_val; n_steps += 1
                    for k, v in m.items():
                        metrics_sum[k] = metrics_sum.get(k, 0) + v
            else:
                loss_val, m = _train_chunked(
                    model, seq, rel, tgt, masks, criterion, optimizer, amp_scaler, use_amp, n_stocks, chunk_size)
                if loss_val is not None:
                    total_loss += loss_val; n_steps += 1
                    for k, v in m.items():
                        metrics_sum[k] = metrics_sum.get(k, 0) + v

        scheduler.step()

        if n_steps > 0:
            total_loss /= n_steps
            for k in metrics_sum:
                metrics_sum[k] /= n_steps

        score = metrics_sum.get('final_score', 0)

        # Save CPU checkpoint for SWA averaging
        if epoch >= swa_start_epoch:
            swa_checkpoints.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        if (epoch + 1) % 5 == 0:
            swa_tag = " [SWA-ckpt]" if epoch >= swa_start_epoch else ""
            n_ckpt = len(swa_checkpoints)
            ckpt_info = f" ckpt={n_ckpt}" if n_ckpt > 0 else ""
            print(f"  [{expert_name}] Epoch {epoch+1:2d}/{num_epochs}{swa_tag} | "
                  f"Loss: {total_loss:.4f} | Score: {score:.4f} | "
                  f"Patience: {patience}/{EARLY_STOPPING_PATIENCE}{ckpt_info}")

        if score > best_score + EARLY_STOPPING_MIN_DELTA:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"  [{expert_name}] Early stop! Best: {best_score:.4f}")
                break

    # Average SWA checkpoints on CPU
    if len(swa_checkpoints) > 0:
        print(f"  Averaging {len(swa_checkpoints)} SWA checkpoints on CPU...")
        swa_state = _average_state_dicts(swa_checkpoints)
        model.load_state_dict(swa_state)
        return best_score
    elif best_state is not None:
        model.load_state_dict(best_state)
        return best_score
    return best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    loss_type = args.loss; num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"SWA Training | Loss: {loss_type} | SWA start: {SWA_START*100:.0f}% | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_swa')
    os.makedirs(output_dir, exist_ok=True)

    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # Override conv_deep: 384 channels OOM on laptop GPU, use 256
    expert_configs = []
    for ec in EXPERT_CONFIGS:
        ec = dict(ec)
        if ec['name'] == 'conv_deep':
            ec['hidden_channels'] = 256
        expert_configs.append(ec)

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
        'stock_embed_dim': STOCK_EMBED_DIM, 'expert_configs': expert_configs,
        'swa': True, 'swa_start': SWA_START,
    }
    with open(os.path.join(output_dir, 'ensemble_config.json'), 'w') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    all_results = {}
    for exp_cfg in expert_configs:
        name = exp_cfg['name']
        model_path = os.path.join(output_dir, f'expert_{name}.pth')
        if os.path.exists(model_path):
            print(f"\n  SKIP {name}: already trained")
            all_results[name] = 0.0
            continue

        print(f"\n{'='*50}")
        print(f"Training: {name} (SWA)")
        model = StockTransformerExpert(n_feats, exp_cfg, num_stocks) if exp_cfg['type'] == 'transformer' \
                else ConvStockExpert(n_feats, exp_cfg, num_stocks)
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
        model.to(device)

        best_score = train_expert_swa(model, exp_cfg, dataset, device, name, loss_type, num_epochs)
        all_results[name] = best_score
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone! SWA")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
