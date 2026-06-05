"""
V5 磁盘管线训练: nhead=8, FFN=4x Transformer 专家
Step 1: 预处理 + Winsorization → 存盘
Step 2: 从磁盘加载 → 训练

用法:
  python code/src/train_v5_disk.py --preprocess   # 仅预处理存盘
  python code/src/train_v5_disk.py --train          # 从磁盘加载训练
  python code/src/train_v5_disk.py                  # 两步一起
"""
import os, sys, json, gc, argparse, shutil
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_v5 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39, create_ranking_dataset_vectorized
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, WeightedRankingLoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)

V3_DIR = './model/v2_ensemble'
PREPROC_DIR = './data/preprocessed_v5'


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


# ============================================================
# Step 1: 预处理 + Winsorization → 存盘
# ============================================================

def winsorize_features(df, feature_cols, lower=0.01, upper=0.99):
    bounds = {}
    for col in feature_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        if hi > lo:
            df[col] = df[col].clip(lo, hi)
            bounds[col] = (float(lo), float(hi))
    return df, bounds


def preprocess_and_save():
    print("=" * 50)
    print("Step 1: 预处理 + 存盘")
    print("=" * 50)

    train_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6)
    train_df['日期'] = pd.to_datetime(train_df['日期'])
    print(f"数据: {len(train_df)} 行, {train_df['日期'].min().date()} ~ {train_df['日期'].max().date()}")

    stock_ids = sorted(train_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)
    print(f"股票: {num_stocks}")

    # 特征工程
    feature_columns = feature_cloums_map[FEATURE_NUM]
    train_df = train_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    min_rows = SEQUENCE_LENGTH + 10
    groups = [g.reset_index(drop=True) for _, g in train_df.groupby('股票代码', sort=False)
              if len(g) >= min_rows]
    print(f"有效分组: {len(groups)}")

    processed = pd.concat([feature_engineer_func_map[FEATURE_NUM](g)
                           for g in tqdm(groups, desc='特征工程')]).reset_index(drop=True)
    del groups, train_df; gc.collect()

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=True)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=feature_columns)

    # Winsorization
    processed, winsor_bounds = winsorize_features(processed, feature_columns, WINSOR_LOWER, WINSOR_UPPER)

    # StandardScaler
    scaler = StandardScaler()
    processed[feature_columns] = scaler.fit_transform(processed[feature_columns])

    # 保存
    os.makedirs(PREPROC_DIR, exist_ok=True)
    processed.to_parquet(os.path.join(PREPROC_DIR, 'features.parquet'), index=False)
    joblib.dump(scaler, os.path.join(PREPROC_DIR, 'scaler.pkl'))
    joblib.dump(stockid2idx, os.path.join(PREPROC_DIR, 'stockid2idx.pkl'))
    joblib.dump(feature_columns, os.path.join(PREPROC_DIR, 'feature_columns.pkl'))

    with open(os.path.join(PREPROC_DIR, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f, indent=2)

    with open(os.path.join(PREPROC_DIR, 'info.json'), 'w') as f:
        json.dump({'num_stocks': num_stocks, 'feature_dim': len(feature_columns)}, f)

    print(f"\n已保存到 {PREPROC_DIR}/")
    print(f"  特征: {len(feature_columns)} 维, 股票: {num_stocks}")
    del processed; gc.collect()


# ============================================================
# Step 2: 从磁盘加载 → 训练
# ============================================================

def train_from_disk():
    print("=" * 50)
    print("Step 2: 从磁盘加载 → 训练")
    print("=" * 50)

    # 加载预处理数据
    processed = pd.read_parquet(os.path.join(PREPROC_DIR, 'features.parquet'))
    scaler = joblib.load(os.path.join(PREPROC_DIR, 'scaler.pkl'))
    stockid2idx = joblib.load(os.path.join(PREPROC_DIR, 'stockid2idx.pkl'))
    feature_columns = joblib.load(os.path.join(PREPROC_DIR, 'feature_columns.pkl'))
    with open(os.path.join(PREPROC_DIR, 'info.json')) as f:
        info = json.load(f)

    print(f"加载: {len(processed)} 行, {info['feature_dim']} 特征, {info['num_stocks']} 股票")

    stock_ids = sorted(stockid2idx.keys())
    num_stocks = len(stock_ids)

    # 构建数据集
    print("构建数据集...")
    proc_sorted = processed.sort_values(['日期', 'instrument']).reset_index(drop=True)
    del processed; gc.collect()

    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        proc_sorted, feature_columns, SEQUENCE_LENGTH)
    print(f"训练天数: {len(train_seq)}")
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del proc_sorted; gc.collect()

    # 训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"\n设备: {device}")
    set_seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shutil.copy2(os.path.join(PREPROC_DIR, 'scaler.pkl'), os.path.join(OUTPUT_DIR, 'scaler.pkl'))

    # 训练 Transformer 专家
    transformer_cfgs = [e for e in EXPERT_CONFIGS if e['type'] == 'transformer']
    results = {}

    for exp_cfg in transformer_cfgs:
        name = exp_cfg['name']
        n_feats = len(feature_columns)
        print(f"\n{'='*50}")
        print(f"训练: {name}")
        print(f"  d_model={exp_cfg['d_model']}, nhead={exp_cfg['nhead']}, "
              f"layers={exp_cfg['num_layers']}, FFN={exp_cfg['dim_feedforward']}")

        model = StockTransformerExpert(n_feats, exp_cfg, num_stocks)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {n_params:,}")
        model.to(device)

        best_score = _train_one_expert(model, exp_cfg, dataset, device, name)
        results[name] = best_score

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'expert_{name}.pth'))
        print(f"  已保存: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 复制 Conv 专家
    print(f"\n复制 Conv 专家...")
    for exp_cfg in EXPERT_CONFIGS:
        if exp_cfg['type'] == 'conv':
            name = exp_cfg['name']
            src = os.path.join(V3_DIR, f'expert_{name}.pth')
            dst = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
            if os.path.exists(src):
                shutil.copy2(src, dst)
                results[name] = exp_cfg.get('train_score', 0.1)
                print(f"  {name}: 已复制")

    # 保存配置
    config_out = {
        'feature_dim': n_feats,
        'num_stocks': num_stocks,
        'expert_configs': EXPERT_CONFIGS,
        'expert_results': results,
    }
    with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_out, f, ensure_ascii=False, indent=2)

    print(f"\nV5 训练完成!")
    for n, s in sorted(results.items()):
        print(f"  {n}: {s:.4f}")


# ============================================================
# 训练引擎 (复用 train_v2.py 逻辑)
# ============================================================

def _train_one_expert(model, exp_cfg, train_dataset, device, expert_name):
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

        for batch in loader:
            seq = batch['sequences'].to(device, non_blocking=pin_memory)
            rel = batch['relevance'].to(device, non_blocking=pin_memory)
            tgt = batch['targets'].to(device, non_blocking=pin_memory)
            masks = batch['masks'].to(device, non_blocking=pin_memory)
            n_stocks = seq.size(1)

            optimizer.zero_grad()
            chunk_losses, chunk_metrics = [], []
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
                        amp_scaler.scale(loss_c).backward()
                    else:
                        loss_c.backward()
                    chunk_losses.append(loss_c.item())
                    with torch.no_grad():
                        mc = calculate_ranking_metrics(masked_c.detach(), tgt_c * mask_c, mask_c, k=5)
                    chunk_metrics.append(mc)

            if chunk_losses:
                if use_amp:
                    amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if use_amp:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                total_loss += np.mean(chunk_losses)
                n_steps += 1

        scheduler.step()
        if n_steps > 0:
            total_loss /= n_steps

        score = -total_loss
        if score > best_score + EARLY_STOPPING_MIN_DELTA:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{expert_name}] Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Loss: {total_loss:.4f} | Score: {score:.4f} | 耐心: {patience}/{EARLY_STOPPING_PATIENCE}")

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"  [{expert_name}] 早停 @ epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score


# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    run_both = not args.preprocess and not args.train

    if run_both or args.preprocess:
        preprocess_and_save()
        print("\n预处理完成！现在重启 Python 并运行:")
        print("  python code/src/train_v5_disk.py --train")

    if run_both:
        print("\n" + "=" * 50)
        input("按 Enter 继续训练（建议先关掉浏览器/IDE 释放内存）...")

    if run_both or args.train:
        if not os.path.exists(os.path.join(PREPROC_DIR, 'features.parquet')):
            print("错误: 未找到预处理数据，请先运行 --preprocess")
            sys.exit(1)
        train_from_disk()
