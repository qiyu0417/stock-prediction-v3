"""
V7 训练: 100 epochs + 训练全部 4 专家 (复用 V5 预处理数据)
用法: python code/src/train_v7.py
"""
import os, sys, json, gc
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np, pandas as pd, torch, joblib
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_v7 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import create_ranking_dataset_vectorized
from train import (
    feature_cloums_map, set_seed, WeightedRankingLoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)

PREPROC_DIR = './data/preprocessed_v5'


def _train_step(model, seq, rel, tgt, masks, criterion, optimizer, scaler, use_amp):
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', enabled=use_amp):
        scores = model(seq)
        masked = scores * masks + (1 - masks) * (-1e9)
        loss = None
        B = seq.size(0)
        for i in range(B):
            valid_idx = masks[i].nonzero().squeeze()
            if valid_idx.numel() <= 1:
                continue
            if valid_idx.dim() == 0:
                valid_idx = valid_idx.unsqueeze(0)
            valid_pred = masked[i][valid_idx]
            valid_rel = rel[i][valid_idx].float()
            loss_i = criterion(valid_pred.unsqueeze(0), valid_rel.unsqueeze(0))
            loss = loss + loss_i if loss is not None else loss_i
    if loss is None:
        return None, None
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
    with torch.no_grad():
        m = calculate_ranking_metrics(masked.detach(), tgt * masks, masks, k=5)
    return loss.item(), m


def _train_chunked(model, seq, rel, tgt, masks, criterion, optimizer, scaler,
                   use_amp, n_stocks, chunk_size):
    chunk_losses, chunk_metrics = [], []
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
            loss_c = loss_c / n_chunks
            if use_amp:
                scaler.scale(loss_c).backward()
            else:
                loss_c.backward()
            chunk_losses.append(loss_c.item())
            with torch.no_grad():
                mc = calculate_ranking_metrics(masked_c.detach(), tgt_c * mask_c, mask_c, k=5)
            chunk_metrics.append(mc)

    if not chunk_losses:
        return None, None
    if use_amp:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    merged_m = {}
    for mc in chunk_metrics:
        for k, v in mc.items():
            merged_m[k] = merged_m.get(k, 0) + v / len(chunk_metrics)
    return np.mean(chunk_losses), merged_m


def train_expert(model, exp_cfg, train_dataset, device, expert_name):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pin_memory = device.type == 'cuda'
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory)

    criterion = WeightedRankingLoss(
        k=5, temperature=1.0, weight_factor=TOP5_WEIGHT,
        pairwise_weight=PAIRWISE_WEIGHT, base_weight=BASE_WEIGHT)
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

    for epoch in range(NUM_EPOCHS):
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
                    model, seq, rel, tgt, masks, criterion, optimizer, amp_scaler,
                    use_amp, n_stocks, chunk_size)
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
        if score > best_score + EARLY_STOPPING_MIN_DELTA:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{expert_name}] Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Loss: {total_loss:.4f} | Score: {score:.4f} | "
                  f"patience: {patience}/{EARLY_STOPPING_PATIENCE}")

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"  [{expert_name}] early stop @ epoch {epoch+1}, best: {best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score if best_state is not None else -float('inf')


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device: {device}")
    set_seed(42)

    print(f"\nload preprocessed data from {PREPROC_DIR}/...")
    processed = pd.read_parquet(os.path.join(PREPROC_DIR, 'features.parquet'))
    scaler = joblib.load(os.path.join(PREPROC_DIR, 'scaler.pkl'))
    stockid2idx = joblib.load(os.path.join(PREPROC_DIR, 'stockid2idx.pkl'))
    feature_columns = joblib.load(os.path.join(PREPROC_DIR, 'feature_columns.pkl'))
    with open(os.path.join(PREPROC_DIR, 'info.json')) as f:
        info = json.load(f)
    print(f"  {len(processed)} rows, {info['feature_dim']} features, {info['num_stocks']} stocks")

    n_feats = len(feature_columns)
    num_stocks = info['num_stocks']

    print("build dataset...")
    proc_sorted = processed.sort_values(['日期', 'instrument']).reset_index(drop=True)
    del processed; gc.collect()

    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        proc_sorted, feature_columns, SEQUENCE_LENGTH)
    print(f"  {len(train_seq)} training days")
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del proc_sorted; gc.collect()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # copy scaler and winsor_bounds
    import shutil
    shutil.copy2(os.path.join(PREPROC_DIR, 'scaler.pkl'), os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    shutil.copy2(os.path.join(PREPROC_DIR, 'winsor_bounds.json'), os.path.join(OUTPUT_DIR, 'winsor_bounds.json'))

    results = {}

    # train ALL 4 experts
    for exp_cfg in EXPERT_CONFIGS:
        name = exp_cfg['name']
        print(f"\n{'='*50}")
        print(f"train: {name} ({exp_cfg['type']})")
        if exp_cfg['type'] == 'transformer':
            print(f"  d_model={exp_cfg['d_model']}, nhead={exp_cfg['nhead']}, "
                  f"layers={exp_cfg['num_layers']}, FFN={exp_cfg['dim_feedforward']}")
        else:
            print(f"  hidden_channels={exp_cfg['hidden_channels']}")

        if exp_cfg['type'] == 'transformer':
            model = StockTransformerExpert(n_feats, exp_cfg, num_stocks)
        else:
            model = ConvStockExpert(n_feats, exp_cfg, num_stocks)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        model.to(device)

        best_score = train_expert(model, exp_cfg, dataset, device, name)
        results[name] = best_score

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'expert_{name}.pth'))
        print(f"  saved: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    config_out = {
        'feature_dim': n_feats,
        'num_stocks': num_stocks,
        'expert_configs': EXPERT_CONFIGS,
        'expert_results': {k: float(v) for k, v in results.items()},
    }
    with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_out, f, ensure_ascii=False, indent=2)

    print(f"\nV7 done!")
    for n, s in sorted(results.items()):
        print(f"  {n}: {s:.4f}")


if __name__ == '__main__':
    main()
