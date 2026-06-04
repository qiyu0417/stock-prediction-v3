"""
V2 Phase 2 训练: 5 专家 (2 Transformer + 2 Conv + 1 Seasonal)
- 数据集构建一次，所有专家复用
- V1 超参数: LR=1e-5, WD=5e-5, GradNorm=3.0, Patience=6
"""
import os, sys, json, gc, multiprocessing as mp
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_v2 import *
from ensemble_models import (
    StockTransformerExpert, ConvStockExpert, MonthSeasonalExpert
)
from utils import engineer_features_158plus39, create_ranking_dataset_vectorized
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, WeightedRankingLoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)


def create_expert(expert_cfg, input_dim, num_stocks):
    t = expert_cfg.get('type', 'transformer')
    if t == 'transformer':
        return StockTransformerExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'conv':
        return ConvStockExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'month_seasonal':
        return MonthSeasonalExpert(input_dim, expert_cfg, num_stocks)
    raise ValueError(f"未知专家类型: {t}")


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)


feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_data(df, stockid2idx):
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    latest = df['日期'].max()
    cutoff = latest - pd.DateOffset(years=2)
    df = df[df['日期'] >= cutoff].copy()
    print(f"  数据范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")
    min_rows = SEQUENCE_LENGTH + 10
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False)
              if len(g) >= min_rows]
    del df; gc.collect()
    print(f"  有效股票: {len(groups)}")

    processed = pd.concat([feature_engineer(g) for g in tqdm(groups, desc='特征工程')]).reset_index(drop=True)
    del groups; gc.collect()

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=True)

    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=feature_columns)

    scaler = StandardScaler()
    processed[feature_columns] = scaler.fit_transform(processed[feature_columns])

    return processed, feature_columns, scaler


def train_expert(model, exp_cfg, train_dataset, device, expert_name):
    """训练单个专家 — 接收预构建的 DataLoader"""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pin_memory = device.type == 'cuda'
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory)

    criterion = WeightedRankingLoss(
        k=5, temperature=1.0, weight_factor=TOP5_WEIGHT,
        pairwise_weight=PAIRWISE_WEIGHT, base_weight=BASE_WEIGHT
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=LEARNING_RATE * 0.005
    )

    use_amp = USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
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
                    model, seq, rel, tgt, masks, criterion, optimizer, scaler, use_amp
                )
                if loss_val is not None:
                    total_loss += loss_val; n_steps += 1
                    for k, v in m.items():
                        metrics_sum[k] = metrics_sum.get(k, 0) + v
            else:
                loss_val, m = _train_chunked(
                    model, seq, rel, tgt, masks, criterion, optimizer, scaler,
                    use_amp, n_stocks, chunk_size
                )
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

        if (epoch + 1) % 5 == 0:
            print(f"  [{expert_name}] Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Loss: {total_loss:.4f} | Score: {score:.4f} | "
                  f"耐心: {patience}/{EARLY_STOPPING_PATIENCE}")

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"  [{expert_name}] 早停! 最佳: {best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score


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


def main():
    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"设备: {gpu_name} ({vram_gb:.1f} GB VRAM), AMP={'ON' if USE_AMP else 'OFF'}")
        print(f"  股票分块: max {MAX_STOCKS_PER_CHUNK}/chunk")
    else:
        device = torch.device('cpu')
        print(f"设备: CPU")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print(f"\n加载数据: {TRAIN_FILE}")
    df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    print(f"行数: {len(df)}, 日期: {df['日期'].min().date()} ~ {df['日期'].max().date()}")

    stock_ids = sorted(df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)
    print(f"股票数: {num_stocks}")

    # 2. 特征工程 + 数据集构建（只做一次!）
    print(f"\n特征工程 ({FEATURE_NUM}, {INPUT_DIM}维)...")
    processed, features, scaler = preprocess_data(df, stockid2idx)
    print(f"处理后样本: {len(processed)}")

    print(f"\n构建排名数据集（只构建一次，两个专家复用）...")
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, features, SEQUENCE_LENGTH
    )
    print(f"训练样本: {len(train_seq)}")
    train_dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)

    # 3. 训练专家
    print(f"\n{'='*60}")
    print(f"训练 {len(EXPERT_CONFIGS)} 个专家")
    print(f"{'='*60}")

    results = {}
    for idx, exp_cfg in enumerate(EXPERT_CONFIGS):
        name = exp_cfg['name']
        save_path = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
        if os.path.exists(save_path):
            print(f"\n── 专家 [{idx+1}/{len(EXPERT_CONFIGS)}]: {name} (已存在，跳过)")
            continue
        print(f"\n── 专家 [{idx+1}/{len(EXPERT_CONFIGS)}]: {name}")
        exp_type = exp_cfg.get('type', 'transformer')
        if exp_type == 'transformer':
            print(f"  架构: d_model={exp_cfg['d_model']}, layers={exp_cfg['num_layers']}, "
                  f"nhead={exp_cfg['nhead']}, ff={exp_cfg['dim_feedforward']}")
        elif exp_type == 'conv':
            print(f"  架构: hidden_channels={exp_cfg['hidden_channels']}, "
                  f"nhead={exp_cfg['nhead']}")
        elif exp_type == 'month_seasonal':
            print(f"  架构: d_model={exp_cfg['d_model']} (月份季节性)")
        else:
            print(f"  架构: {exp_type}")

        model = create_expert(exp_cfg, INPUT_DIM, num_stocks).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  参数量: {n_params:,} 类型: {exp_cfg.get('type', 'transformer')}")

        score = train_expert(model, exp_cfg, train_dataset, device, name)
        results[name] = score

        torch.save(model.state_dict(), save_path)
        print(f"  >> {name}: best_score={score:.4f}, 已保存")

    # 4. 保存配置
    final_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'feature_num': FEATURE_NUM,
        'input_dim': INPUT_DIM,
        'expert_configs': EXPERT_CONFIGS,
        'expert_results': results,
        'num_stocks': num_stocks,
        'stockid2idx': stockid2idx,
        'feature_list': features,
        'mc_samples': MC_SAMPLES,
        'expert_weights': [1.0 / len(EXPERT_CONFIGS)] * len(EXPERT_CONFIGS),
    }
    with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json'), 'w') as f:
        json.dump(final_config, f, indent=2, ensure_ascii=False)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

    print(f"\n{'='*60}")
    print("V2 训练完成!")
    print(f"{'='*60}")
    for name, score in results.items():
        print(f"  {name:20s}: {score:.6f}")
    print(f"输出: {OUTPUT_DIR}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
