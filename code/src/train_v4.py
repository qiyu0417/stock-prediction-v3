"""
V4 训练: MetaAggregator (基于已有 V2 专家)
- 加载 5 个 V2 预训练专家
- 收集全量训练数据上的专家预测 (MC Dropout)
- 训练 MetaAggregator 学习最优融合权重
- 保存到 model/v4_ensemble/
"""
import os, sys, json, gc, multiprocessing as mp
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble_models import (
    StockTransformerExpert, ConvStockExpert, MonthSeasonalExpert, MetaAggregator
)
from utils import engineer_features_158plus39, create_ranking_dataset_vectorized
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed,
    RankingDataset, collate_fn
)
# Use V2 config for expert architecture, V3 config for MetaAggregator params
from config_v2 import (
    SEQUENCE_LENGTH, FEATURE_NUM, INPUT_DIM, NUM_STOCKS,
    EXPERT_CONFIGS, BATCH_SIZE, USE_AMP, MAX_STOCKS_PER_CHUNK
)
from config_v4 import META_HIDDEN_DIM, META_EPOCHS, META_LR

OUTPUT_DIR = './model/v4_ensemble'
V2_DIR = './model/v2_ensemble'
DATA_PATH = './data'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)

feature_engineer_func_map['158+39'] = _engineer_158plus39


def create_expert(expert_cfg, input_dim, num_stocks):
    t = expert_cfg.get('type', 'transformer')
    if t == 'transformer':
        return StockTransformerExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'conv':
        return ConvStockExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'month_seasonal':
        return MonthSeasonalExpert(input_dim, expert_cfg, num_stocks)
    raise ValueError(f"未知专家类型: {t}")


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

    processed_list = []
    for g in tqdm(groups, desc='特征工程'):
        processed_list.append(feature_engineer(g))
    processed = pd.concat(processed_list).reset_index(drop=True)
    del groups, processed_list; gc.collect()

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=True)

    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=feature_columns)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    processed[feature_columns] = scaler.fit_transform(processed[feature_columns])

    return processed, feature_columns, scaler


def main():
    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"设备: CPU")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print(f"\n加载数据: {TRAIN_FILE}")
    df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    print(f"行数: {len(df)}")

    stock_ids = sorted(df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)
    print(f"股票数: {num_stocks}")

    # 2. 特征工程 (复用 V2 scaler 以保证一致性)
    print(f"\n特征工程 ({FEATURE_NUM}, {INPUT_DIM}维)...")
    processed, features, _ = preprocess_data(df, stockid2idx)

    scaler_path = os.path.join(V2_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        common = [c for c in scaler.feature_names_in_ if c in processed.columns]
        processed[common] = scaler.transform(processed[common])
        features = common
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
        print(f"  使用 V2 scaler, 特征数: {len(features)}")

    print(f"处理后样本: {len(processed)}")

    # 3. 构建数据集
    print(f"\n构建排名数据集...")
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, features, SEQUENCE_LENGTH
    )
    print(f"训练样本: {len(train_seq)} 天")
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)

    # 4. 加载 V2 专家 (CPU 上加载，逐个移到 GPU 推理)
    print(f"\n加载 {len(EXPERT_CONFIGS)} 个 V2 专家 (CPU)...")
    experts_cpu = []
    for cfg in EXPERT_CONFIGS:
        name = cfg['name']
        path = os.path.join(V2_DIR, f'expert_{name}.pth')
        if not os.path.exists(path):
            print(f"  警告: 未找到 {path}, 跳过")
            continue
        model = create_expert(cfg, len(features), num_stocks)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        experts_cpu.append((name, model))
        print(f"  已加载: {name} (CPU)")

    if len(experts_cpu) < 2:
        raise RuntimeError(f"至少需要2个专家, 当前: {len(experts_cpu)}")

    # 5. 收集专家预测 (逐个专家移到 GPU 推理)
    print(f"\n收集专家预测 (MC Dropout, 5 次前向)...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    num_samples = len(dataset)
    num_experts = len(experts_cpu)
    all_expert_scores = [None] * num_samples
    all_targets = [None] * num_samples
    all_masks = [None] * num_samples
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999

    for exp_idx, (exp_name, expert_cpu) in enumerate(experts_cpu):
        print(f"\n  专家 [{exp_idx+1}/{num_experts}]: {exp_name}")
        expert = expert_cpu.to(device)
        expert.train()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"  {exp_name}", total=num_samples)):
                seq = batch['sequences'].to(device)

                mc_scores = []
                for _ in range(5):
                    if seq.size(1) <= chunk_size:
                        s = expert(seq)
                    else:
                        chunk_scores = []
                        for start in range(0, seq.size(1), chunk_size):
                            end = min(start + chunk_size, seq.size(1))
                            sc = expert(seq[:, start:end].contiguous())
                            chunk_scores.append(sc)
                        s = torch.cat(chunk_scores, dim=1)
                    mc_scores.append(s)
                avg = torch.stack(mc_scores).mean(dim=0).cpu()

                if all_expert_scores[batch_idx] is None:
                    all_expert_scores[batch_idx] = []
                all_expert_scores[batch_idx].append(avg)

                if exp_idx == 0:
                    all_targets[batch_idx] = batch['targets']
                    all_masks[batch_idx] = batch['masks']

        # 释放 GPU 内存
        expert = expert.cpu()
        del expert
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 堆叠为 [B, N, E] 格式
    expert_stacks = []
    for batch_idx in range(num_samples):
        stack = torch.stack(all_expert_scores[batch_idx], dim=-1)
        expert_stacks.append(stack)
    all_expert_scores = expert_stacks

    print(f"收集了 {len(all_expert_scores)} 个样本")

    # 6. 训练 MetaAggregator
    print(f"\n训练 MetaAggregator ({META_EPOCHS} epochs)...")
    meta = MetaAggregator(len(experts_cpu), num_stocks, hidden_dim=META_HIDDEN_DIM).to(device)

    meta_optimizer = torch.optim.Adam(meta.parameters(), lr=META_LR)
    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        meta_optimizer, T_0=META_EPOCHS // 3, T_mult=2, eta_min=META_LR * 0.01
    )

    for epoch in range(META_EPOCHS):
        meta.train()
        total_loss, n_batches = 0, 0
        indices = list(range(len(all_expert_scores)))
        np.random.shuffle(indices)

        for i in indices:
            es = all_expert_scores[i].to(device)
            tgt = all_targets[i].to(device)
            m = all_masks[i].to(device)

            meta_optimizer.zero_grad()
            final = meta(es)
            masked_final = final * m + (1 - m) * (-1e9)

            batch_loss = None
            B = es.size(0)
            for j in range(B):
                valid_idx = m[j].nonzero().squeeze()
                if valid_idx.numel() <= 1:
                    continue
                if valid_idx.dim() == 0:
                    valid_idx = valid_idx.unsqueeze(0)
                valid_pred = masked_final[j][valid_idx]
                valid_tgt = tgt[j][valid_idx]
                if len(valid_pred) > 1:
                    _, sorted_idx = torch.sort(valid_tgt, descending=True)
                    rel = torch.zeros_like(valid_tgt)
                    rel[sorted_idx] = torch.arange(len(valid_tgt), 0, -1,
                                                   device=device, dtype=torch.float32)
                    loss = F.mse_loss(valid_pred, rel)
                    batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                (batch_loss / B).backward()
                meta_optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

        meta_scheduler.step()
        if n_batches > 0 and (epoch + 1) % 5 == 0:
            print(f"  [MetaAggregator] Epoch {epoch+1}/{META_EPOCHS} Loss: {total_loss/max(n_batches,1):.4f}")

    # 7. 保存
    torch.save(meta.state_dict(), os.path.join(OUTPUT_DIR, 'meta_aggregator.pth'))

    # 复制专家权重到 V3 目录
    import shutil
    for cfg in EXPERT_CONFIGS:
        src = os.path.join(V2_DIR, f"expert_{cfg['name']}.pth")
        dst = os.path.join(OUTPUT_DIR, f"expert_{cfg['name']}.pth")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    final_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'feature_num': FEATURE_NUM,
        'input_dim': INPUT_DIM,
        'expert_configs': EXPERT_CONFIGS,
        'num_stocks': num_stocks,
        'stockid2idx': stockid2idx,
        'feature_list': features,
        'mc_samples': 20,
    }
    with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json'), 'w') as f:
        json.dump(final_config, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("MetaAggregator 训练完成!")
    print(f"专家数: {len(experts_cpu)}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
