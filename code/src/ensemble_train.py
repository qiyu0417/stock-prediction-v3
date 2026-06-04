"""
集成学习训练脚本
- 滚动窗口数据分割 (随机2月训练+1月验证)
- 多类型专家: Transformer / 卷积(TCN) / 对抗学习
- Adam + 余弦退火热重启 优化
- MC Dropout + 随机深度
- 元调度器训练
"""
import os
import sys
import json
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import joblib

from ensemble_config import *
from ensemble_models import (
    StockTransformerExpert, ConvStockExpert, AdversarialStockExpert,
    MonthSeasonalExpert, MetaAggregator, EnsemblePredictor
)
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized

# 原始项目的特征列映射和工具函数
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, WeightedRankingLoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)


# ============================================================
# 模型工厂
# ============================================================
def create_expert_model(expert_cfg, input_dim, num_stocks):
    """根据配置创建对应类型的专家模型"""
    exp_type = expert_cfg.get('type', 'transformer')

    if exp_type == 'transformer':
        return StockTransformerExpert(input_dim, expert_cfg, num_stocks)

    elif exp_type == 'conv':
        return ConvStockExpert(input_dim, expert_cfg, num_stocks)

    elif exp_type == 'adversarial':
        base_type = expert_cfg.get('base_type', 'transformer')
        if base_type == 'transformer':
            base_expert = StockTransformerExpert(input_dim, expert_cfg, num_stocks)
        else:
            base_expert = ConvStockExpert(input_dim, expert_cfg, num_stocks)
        d_model = expert_cfg.get('d_model', expert_cfg.get('hidden_channels', 256))
        return AdversarialStockExpert(
            base_expert, d_model,
            num_time_domains=expert_cfg.get('num_time_domains', 12),
            adv_lambda=expert_cfg.get('adv_lambda', 0.1)
        )

    elif exp_type == 'month_seasonal':
        return MonthSeasonalExpert(input_dim, expert_cfg, num_stocks)

    else:
        raise ValueError(f"未知专家类型: {exp_type}")


# ============================================================
# 滚动窗口数据分割
# ============================================================
def create_rolling_window_splits(df, sequence_length):
    """
    滚动窗口分割: 3个月一个窗口，随机取2个月训练+1个月验证
    返回: [(train_df, val_df, window_info), ...]
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df['year_month'] = df['日期'].dt.to_period('M')

    all_months = sorted(df['year_month'].unique())
    print(f"数据覆盖月份: {all_months[0]} ~ {all_months[-1]}, 共 {len(all_months)} 个月")

    if len(all_months) < WINDOW_SIZE_MONTHS:
        raise ValueError(f"数据不足: 至少需要 {WINDOW_SIZE_MONTHS} 个月")

    # 创建所有3个月滑动窗口
    windows = []
    for i in range(len(all_months) - WINDOW_SIZE_MONTHS + 1):
        window_months = all_months[i:i + WINDOW_SIZE_MONTHS]
        windows.append(window_months)

    print(f"共创建 {len(windows)} 个滑动窗口")

    rng = random.Random(RANDOM_SPLIT_SEED)
    splits = []

    for win_idx, window_months in enumerate(windows):
        indices = list(range(WINDOW_SIZE_MONTHS))
        rng.shuffle(indices)

        val_indices = set(indices[:VAL_MONTHS])
        train_months = [m for j, m in enumerate(window_months) if j not in val_indices]
        val_months = [m for j, m in enumerate(window_months) if j in val_indices]

        train_mask = df['year_month'].isin(train_months)
        val_mask = df['year_month'].isin(val_months)

        train_df = df[train_mask].copy()
        val_data = df[val_mask].copy()

        # 验证集保留 sequence_length-1 天的序列上下文
        if len(val_data) > 0:
            val_min_date = val_data['日期'].min()
            context_start = val_min_date - pd.tseries.offsets.BDay(sequence_length)
            context_mask = (df['日期'] >= context_start) & (df['日期'] < val_min_date)
            val_with_context = pd.concat([df[context_mask].copy(), val_data])
        else:
            val_with_context = val_data

        # 训练集也需要前 sequence_length-1 个交易日作为序列上下文
        if len(train_df) > 0:
            train_min_date = train_df['日期'].min()
            train_context_start = train_min_date - pd.tseries.offsets.BDay(sequence_length)
            train_context = df[(df['日期'] >= train_context_start) & (df['日期'] < train_min_date)].copy()
            train_df_with_context = pd.concat([train_context, train_df])
        else:
            train_df_with_context = train_df

        splits.append({
            'train_df': train_df_with_context,
            'val_df': val_with_context,
            'train_months': [str(m) for m in train_months],
            'val_months': [str(m) for m in val_months],
            'window_idx': win_idx,
            'val_start_date': str(val_data['日期'].min().date()) if len(val_data) > 0 else None,
        })

    return splits


# ============================================================
# 数据预处理（单窗口）
# ============================================================
def preprocess_window(train_df, val_df, stockid2idx, feature_num):
    """对单个窗口进行特征工程和标准化"""
    assert feature_num in feature_engineer_func_map
    feature_engineer = feature_engineer_func_map[feature_num]
    feature_columns = feature_cloums_map[feature_num]

    def process_one(df, desc):
        df = df.copy()
        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        # 过滤数据不足的股票（至少需要 sequence_length + 缓冲）
        min_rows = SEQUENCE_LENGTH + 10
        groups = []
        for _, group in df.groupby('股票代码', sort=False):
            group = group.reset_index(drop=True)
            if len(group) >= min_rows:
                groups.append(group)
        if len(groups) == 0:
            return None, None

        num_procs = min(8, mp.cpu_count())
        with mp.Pool(processes=num_procs) as pool:
            processed_list = list(tqdm(
                pool.imap(feature_engineer, groups),
                total=len(groups), desc=desc, leave=False
            ))

        processed = pd.concat(processed_list).reset_index(drop=True)
        processed['instrument'] = processed['股票代码'].map(stockid2idx)
        processed = processed.dropna(subset=['instrument']).copy()
        processed['instrument'] = processed['instrument'].astype(np.int64)
        processed = _build_label_and_clean(processed, drop_small_open=True)
        return processed, feature_columns

    train_processed, features = process_one(train_df, "训练集特征工程")
    if train_processed is None:
        return None, None, None, None

    val_processed = None
    if val_df is not None and len(val_df) > 0:
        val_processed, _ = process_one(val_df, "验证集特征工程")

    # 标准化
    train_processed[features] = train_processed[features].replace([np.inf, -np.inf], np.nan)
    train_processed = train_processed.dropna(subset=features)

    if val_processed is not None:
        val_processed[features] = val_processed[features].replace([np.inf, -np.inf], np.nan)
        val_processed = val_processed.dropna(subset=features)

    scaler = StandardScaler()
    train_processed[features] = scaler.fit_transform(train_processed[features])
    if val_processed is not None:
        val_processed[features] = scaler.transform(val_processed[features])

    return train_processed, val_processed, features, scaler


# ============================================================
# 训练单个专家
# ============================================================
def train_expert(model, exp_cfg, train_data, val_data, features, sequence_length,
                 device, expert_name, writer, window_idx=0, scaler=None):
    """训练单个专家模型（支持Transformer/Conv/Adversarial）"""
    exp_type = exp_cfg.get('type', 'transformer')
    is_adversarial = (exp_type == 'adversarial')

    # 构建数据集
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        train_data, features, sequence_length
    )

    val_seq, val_tgt, val_rel, val_stk = None, None, None, None
    if val_data is not None:
        val_seq, val_tgt, val_rel, val_stk = create_ranking_dataset_vectorized(
            val_data, features, sequence_length
        )

    print(f"  [{expert_name}] 训练样本: {len(train_seq)}, "
          f"验证样本: {len(val_seq) if val_seq else 0}")

    if len(train_seq) == 0:
        return None

    train_dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    val_loader = None
    if val_seq and len(val_seq) > 0:
        val_dataset = RankingDataset(val_seq, val_tgt, val_rel, val_stk)
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, num_workers=0, pin_memory=False
        )

    # 损失 & 优化器
    criterion = WeightedRankingLoss(
        k=5, temperature=1.0, weight_factor=TOP5_WEIGHT,
        pairwise_weight=PAIRWISE_WEIGHT, base_weight=BASE_WEIGHT
    )
    # Adam 优化器: 比 AdamW 更利于跳出局部最优
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8
    )
    # 余弦退火+热重启: 周期性提高学习率，跳出局部最优
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=LEARNING_RATE * 0.005
    )

    # 对抗损失权重动态调度
    if is_adversarial:
        adv_lambda_base = exp_cfg.get('adv_lambda', 0.1)

    best_score = -float('inf')
    best_state = None
    patience_counter = 0       # 早停计数器
    stopped_early = False

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_ranking_loss = 0
        train_adv_loss = 0
        train_metrics_sum = {}
        local_step = 0

        # 动态对抗权重: 前期聚焦主任务，后期逐渐加强对抗
        if is_adversarial and ADV_LAMBDA_SCHEDULE:
            adv_lambda = adv_lambda_base * min(1.0, epoch / (NUM_EPOCHS * 0.5))
        elif is_adversarial:
            adv_lambda = adv_lambda_base
        else:
            adv_lambda = 0

        for batch in train_loader:
            seq = batch['sequences'].to(device)      # [B, S, L, F]
            tgt = batch['targets'].to(device)        # [B, S]
            rel = batch['relevance'].to(device)      # [B, S] 放到GPU
            masks = batch['masks'].to(device)        # [B, S]

            optimizer.zero_grad()

            B = seq.size(0)

            # 前向传播
            if is_adversarial:
                scores, features = model(seq, return_features=True)
            else:
                scores = model(seq)

            masked_outputs = scores * masks + (1 - masks) * (-1e9)

            # 排序损失
            rank_loss = None
            for i in range(B):
                valid_idx = masks[i].nonzero().squeeze()
                if valid_idx.numel() == 0:
                    continue
                if valid_idx.dim() == 0:
                    valid_idx = valid_idx.unsqueeze(0)
                valid_pred = masked_outputs[i][valid_idx]
                valid_rel = rel[i][valid_idx].float()
                if len(valid_pred) > 1:
                    loss_i = criterion(valid_pred.unsqueeze(0), valid_rel.unsqueeze(0))
                    rank_loss = rank_loss + loss_i if rank_loss is not None else loss_i

            total_loss = rank_loss if rank_loss is not None else torch.tensor(0.0, device=device)

            # 对抗损失
            if is_adversarial and adv_lambda > 0:
                # 为每个样本分配时间段标签（基于年份+月份分组）
                # 简化: 根据样本索引分配伪时间段标签
                time_labels = torch.randint(0, exp_cfg.get('num_time_domains', 12),
                                            (features.size(0),), device=device)
                adv_loss = model.get_adversarial_loss(features, time_labels)
                total_loss = total_loss + adv_lambda * adv_loss
                train_adv_loss += adv_loss.item()

            if rank_loss is not None:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                train_loss += total_loss.item()
                train_ranking_loss += rank_loss.item()

            # 指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, tgt * masks, masks, k=5)
                for k, v in metrics.items():
                    train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v
            local_step += 1

        scheduler.step()

        if local_step > 0:
            train_loss /= local_step
            for k in train_metrics_sum:
                train_metrics_sum[k] /= local_step

        # 验证
        val_score = None
        if val_loader is not None:
            model.eval()
            val_metrics_sum = {}
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    seq = batch['sequences'].to(device)
                    tgt = batch['targets'].to(device)
                    masks = batch['masks'].to(device)

                    if is_adversarial:
                        scores, _ = model(seq, return_features=True)
                    else:
                        scores = model(seq)
                    masked_outputs = scores * masks + (1 - masks) * (-1e9)

                    metrics = calculate_ranking_metrics(masked_outputs, tgt * masks, masks, k=5)
                    for k, v in metrics.items():
                        val_metrics_sum[k] = val_metrics_sum.get(k, 0) + v
                    val_steps += 1

            if val_steps > 0:
                for k in val_metrics_sum:
                    val_metrics_sum[k] /= val_steps

            val_score = val_metrics_sum.get('final_score', 0)

            if val_score > best_score + EARLY_STOPPING_MIN_DELTA:
                best_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0  # 重置早停计数
            else:
                patience_counter += 1

        # 日志
        if writer and (epoch + 1) % 5 == 0:
            writer.add_scalar(f'expert/{expert_name}/train_loss', train_loss, epoch)
            writer.add_scalar(f'expert/{expert_name}/val_score',
                              val_score if val_score else 0, epoch)
            if is_adversarial:
                writer.add_scalar(f'expert/{expert_name}/adv_loss',
                                  train_adv_loss / max(local_step, 1), epoch)
                writer.add_scalar(f'expert/{expert_name}/adv_lambda', adv_lambda, epoch)

        if (epoch + 1) % 5 == 0:
            adv_str = f" Adv: {train_adv_loss/max(local_step,1):.4f}" if is_adversarial else ""
            early_str = f" [耐心: {patience_counter}/{EARLY_STOPPING_PATIENCE}]" if val_loader else ""
            print(f"  [{expert_name}] Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Loss: {train_loss:.4f}{adv_str}{early_str} | "
                  f"Val Score: {val_score:.4f}" if val_score is not None else "")

        # 早停检查
        if val_loader is not None and patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  [{expert_name}] 早停! Epoch {epoch+1}, "
                  f"连续 {EARLY_STOPPING_PATIENCE} 轮无提升, 最佳分数: {best_score:.4f}")
            stopped_early = True
            break

    # 恢复最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_score


# ============================================================
# 训练元调度器
# ============================================================
def train_meta_aggregator(experts, expert_configs, meta, train_data, features,
                          sequence_length, device, scaler):
    """元调度器: 学习如何加权组合各专家预测"""
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        train_data, features, sequence_length
    )
    if len(train_seq) == 0:
        return

    print(f"元调度器训练样本: {len(train_seq)}")
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )

    # 收集各专家预测
    print("Step 1: 收集所有专家预测...")
    all_expert_scores = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="专家推理"):
            seq = batch['sequences'].to(device)
            masks = batch['masks']

            batch_scores = []
            for i, expert in enumerate(experts):
                # MC Dropout 推理 (减少采样次数以提高速度)
                scores = expert.predict_with_mc_dropout(seq, num_samples=5)
                batch_scores.append(scores)

            expert_stack = torch.stack(batch_scores, dim=-1)  # [B, N, E]
            all_expert_scores.append(expert_stack)
            all_targets.append(batch['targets'])
            all_masks.append(masks)

    # 训练元调度器
    print("Step 2: 训练元调度器...")
    meta_optimizer = torch.optim.Adam(meta.parameters(), lr=META_LR)
    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        meta_optimizer, T_0=META_EPOCHS // 3, T_mult=2, eta_min=META_LR * 0.01
    )

    for epoch in range(META_EPOCHS):
        meta.train()
        total_loss = 0
        n_batches = 0

        for i in range(len(all_expert_scores)):
            expert_scores = all_expert_scores[i].to(device)
            targets = all_targets[i].to(device)
            masks = all_masks[i].to(device)

            meta_optimizer.zero_grad()
            final = meta(expert_scores)
            masked_final = final * masks + (1 - masks) * (-1e9)

            batch_loss = None
            B = expert_scores.size(0)
            for j in range(B):
                valid_idx = masks[j].nonzero().squeeze()
                if valid_idx.numel() <= 1:
                    continue
                if valid_idx.dim() == 0:
                    valid_idx = valid_idx.unsqueeze(0)
                valid_pred = masked_final[j][valid_idx]
                valid_tgt = targets[j][valid_idx]
                if len(valid_pred) > 1:
                    _, sorted_idx = torch.sort(valid_tgt, descending=True)
                    rel = torch.zeros_like(valid_tgt)
                    rel[sorted_idx] = torch.arange(len(valid_tgt), 0, -1,
                                                   device=device, dtype=torch.float32)
                    loss = F.mse_loss(valid_pred, rel)
                    batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                batch_loss = batch_loss / B
                batch_loss.backward()
                meta_optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

        meta_scheduler.step()

        if n_batches > 0 and (epoch + 1) % 5 == 0:
            print(f"  [元调度器] Epoch {epoch+1}/{META_EPOCHS} Loss: {total_loss/max(n_batches,1):.4f}")


# ============================================================
# 主训练流程
# ============================================================
def main():
    set_seed(RANDOM_SEED)

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"设备: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    else:
        device = torch.device('cpu')
    print(f"设备: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print(f"\n{'='*60}")
    print(f"加载数据: {TRAIN_FILE}")
    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    print(f"数据行数: {len(full_df)}")

    # 2. 股票ID映射
    all_stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
    num_stocks = len(stockid2idx)
    print(f"股票数量: {num_stocks}")

    # 3. 滚动窗口分割
    print(f"\n{'='*60}")
    print("滚动窗口分割")
    print(f"{'='*60}")
    window_splits = create_rolling_window_splits(full_df, SEQUENCE_LENGTH)
    print(f"窗口数: {len(window_splits)}")

    # 4. 预处理每个窗口
    print(f"\n{'='*60}")
    print("预处理各窗口")
    print(f"{'='*60}")

    all_window_data = []
    for split_info in window_splits:
        print(f"\n窗口 {split_info['window_idx']}: "
              f"训练={[str(m) for m in split_info['train_months']]}, "
              f"验证={[str(m) for m in split_info['val_months']]}")

        train_processed, val_processed, features, scaler = preprocess_window(
            split_info['train_df'], split_info['val_df'],
            stockid2idx, FEATURE_NUM
        )

        if train_processed is not None and len(train_processed) > 100:
            all_window_data.append({
                **split_info,
                'train_processed': train_processed,
                'val_processed': val_processed,
                'features': features,
                'scaler': scaler,
            })
            print(f"  训练样本: {len(train_processed)}, "
                  f"验证样本: {len(val_processed) if val_processed is not None else 0}")

    if len(all_window_data) == 0:
        raise RuntimeError("无有效窗口数据")

    feature_list = all_window_data[0]['features']
    input_dim = len(feature_list)
    print(f"\n特征数: {input_dim}")

    # 5. 训练各专家
    print(f"\n{'='*60}")
    print(f"训练 {len(EXPERT_CONFIGS)} 个专家模型")
    print(f"  Transformer: {sum(1 for c in EXPERT_CONFIGS if c.get('type','transformer')=='transformer')}")
    print(f"  Conv/TCN:    {sum(1 for c in EXPERT_CONFIGS if c.get('type')=='conv')}")
    print(f"  Adversarial: {sum(1 for c in EXPERT_CONFIGS if c.get('type')=='adversarial')}")
    print(f"{'='*60}")

    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, 'log'))

    experts = []
    expert_results = {}
    expert_scalers = []

    for exp_idx, exp_cfg in enumerate(EXPERT_CONFIGS):
        exp_name = exp_cfg['name']
        exp_type = exp_cfg.get('type', 'transformer')
        print(f"\n{'─'*40}")
        print(f"专家 [{exp_idx+1}/{len(EXPERT_CONFIGS)}]: {exp_name} (类型: {exp_type})")
        print(f"{'─'*40}")

        model = create_expert_model(exp_cfg, input_dim, num_stocks).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  参数量: {n_params:,}")

        best_overall_score = -float('inf')
        best_overall_state = None
        best_scaler = None

        # 跨窗口训练
        for wdata in all_window_data:
            score = train_expert(
                model, exp_cfg,
                wdata['train_processed'],
                wdata['val_processed'],
                feature_list,
                SEQUENCE_LENGTH,
                device, exp_name, writer,
                window_idx=wdata['window_idx'],
                scaler=wdata['scaler']
            )
            if score is not None and score > best_overall_score:
                best_overall_score = score
                best_overall_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_scaler = wdata['scaler']

        if best_overall_state is not None:
            model.load_state_dict(best_overall_state)

        experts.append(model)
        expert_results[exp_name] = best_overall_score
        expert_scalers.append(best_scaler)

        # 保存专家
        torch.save(model.state_dict(),
                   os.path.join(OUTPUT_DIR, f'expert_{exp_name}.pth'))
        print(f"  >> {exp_name} 最佳分数: {best_overall_score:.4f}, 已保存")

    # 6. 训练元调度器
    print(f"\n{'='*60}")
    print("训练元调度器 (Meta Aggregator)")
    print(f"{'='*60}")

    meta_train_data = all_window_data[-1]['train_processed']
    meta = MetaAggregator(len(experts), num_stocks, hidden_dim=META_HIDDEN_DIM).to(device)
    train_meta_aggregator(
        experts, EXPERT_CONFIGS, meta,
        meta_train_data, feature_list,
        SEQUENCE_LENGTH, device, expert_scalers[-1]
    )
    torch.save(meta.state_dict(), os.path.join(OUTPUT_DIR, 'meta_aggregator.pth'))

    # 7. 保存配置
    final_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'feature_num': FEATURE_NUM,
        'input_dim': input_dim,
        'expert_configs': EXPERT_CONFIGS,
        'expert_results': expert_results,
        'num_stocks': num_stocks,
        'stockid2idx': stockid2idx,
        'feature_list': feature_list,
        'mc_samples': MC_SAMPLES,
        'window_count': len(all_window_data),
    }
    with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json'), 'w') as f:
        json.dump(final_config, f, indent=2, ensure_ascii=False)

    # 保存scaler
    if expert_scalers[-1] is not None:
        joblib.dump(expert_scalers[-1], os.path.join(OUTPUT_DIR, 'scaler.pkl'))

    print(f"\n{'='*60}")
    print("集成训练完成!")
    print(f"{'='*60}")
    print(f"各专家最终分数:")
    for name, score in expert_results.items():
        print(f"  {name:30s}: {score:.6f}" if score else f"  {name:30s}: N/A")
    print(f"输出目录: {OUTPUT_DIR}")

    writer.close()
    return expert_results


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
