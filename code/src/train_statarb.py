"""
Train StatArbRegressionExpert — Neural GARCH + mean reversion signal.
Orthogonal to existing momentum/trend models.
"""
import os, sys, json, gc, argparse
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np, pandas as pd, torch, joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble_models import StatArbRegressionExpert
from utils import engineer_features_158plus39, create_ranking_dataset_vectorized
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed,
    calculate_ranking_metrics, RankingDataset, collate_fn,
    HybridRankingLoss
)

# ── Config ──
SEQUENCE_LENGTH = 60
FEATURE_NUM = '158+39'
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-5
MAX_GRAD_NORM = 3.0
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 2e-4
USE_AMP = True
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# 3 StatArb variants for ensemble diversity
EXPERT_CONFIGS = [
    {
        'name': 'statarb_base',
        'type': 'statarb',
        'd_model': 192,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 384,
        'dropout': 0.12,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.85,
    },
    {
        'name': 'statarb_deep',
        'type': 'statarb',
        'd_model': 192,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 384,
        'dropout': 0.15,
        'mc_dropout_rate': 0.12,
        'sd_prob': 0.80,
    },
    {
        'name': 'statarb_wide',
        'type': 'statarb',
        'd_model': 256,
        'nhead': 8,
        'num_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.12,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.85,
    },
]


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


def train_expert(model, exp_cfg, train_dataset, device, expert_name, num_epochs=None):
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pin_memory = device.type == 'cuda'
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory)

    criterion = HybridRankingLoss(k=5)
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


def winsorize_features(df, feature_cols, lower=0.01, upper=0.99):
    bounds = {}
    for i, col in enumerate(feature_cols):
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        if hi > lo:
            df[col] = df[col].clip(lo, hi)
            bounds[col] = (float(lo), float(hi))
        if i % 50 == 0:
            gc.collect()
    return df, bounds


def preprocess_with_winsor(df, stockid2idx, winsor_bounds=None):
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    min_rows = SEQUENCE_LENGTH + 10
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False)
              if len(g) >= min_rows]
    del df
    gc.collect()
    print(f"  Valid stocks: {len(groups)}")

    processed = pd.concat([feature_engineer(g) for g in tqdm(groups, desc='  Feature engineering')]).reset_index(drop=True)
    del groups
    gc.collect()

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=True)

    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=feature_columns)

    if winsor_bounds is None:
        processed, winsor_bounds = winsorize_features(processed, feature_columns, WINSOR_LOWER, WINSOR_UPPER)
    else:
        for col, (lo, hi) in winsor_bounds.items():
            if col in processed.columns:
                processed[col] = processed[col].clip(lo, hi)

    scaler = StandardScaler()
    processed[feature_columns] = scaler.fit_transform(processed[feature_columns])

    return processed, feature_columns, scaler, winsor_bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    num_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"StatArb Training | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', 'stock_emb_8_statarb')
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading data...")
    full_df = pd.read_csv('./data/train.csv', dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    print(f"  Total: {len(full_df)} rows, {full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}")

    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)
    print(f"  Stocks: {num_stocks}")

    print("\nPreprocessing...")
    processed, features, scaler, winsor_bounds = preprocess_with_winsor(full_df, stockid2idx)
    n_feats = len(features)
    print(f"  Feature dim: {n_feats}")

    del full_df
    gc.collect()

    print("\nBuilding ranking dataset...")
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, features, SEQUENCE_LENGTH)
    print(f"  Training days: {len(train_seq)}")
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)

    del processed, train_tgt, train_rel, train_stk
    gc.collect()

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    with open(os.path.join(output_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f)
    print(f"Saved scaler + winsor_bounds -> {output_dir}/")

    all_results = {}

    for exp_cfg in EXPERT_CONFIGS:
        name = exp_cfg['name']
        model_path = os.path.join(output_dir, f'expert_{name}.pth')
        if os.path.exists(model_path):
            print(f"\n  SKIP {name}: already trained")
            all_results[name] = 0.0
            continue

        print(f"\n{'=' * 50}")
        print(f"Training: {name} (StatArb)")
        print(f"  d={exp_cfg['d_model']}, layers={exp_cfg['num_layers']}, "
              f"FFN={exp_cfg['dim_feedforward']}")

        model = StatArbRegressionExpert(n_feats, exp_cfg, num_stocks)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        model.to(device)

        best_score = train_expert(model, exp_cfg, dataset, device, name, num_epochs)
        all_results[name] = best_score

        torch.save(model.state_dict(), os.path.join(output_dir, f'expert_{name}.pth'))
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    config_out = {
        'feature_dim': n_feats,
        'num_stocks': num_stocks,
        'expert_configs': EXPERT_CONFIGS,
        'expert_results': {k: float(v) for k, v in all_results.items()},
        'features': FEATURE_NUM,
        'loss_type': 'hybrid',
        'model_type': 'statarb',
    }
    with open(os.path.join(output_dir, 'ensemble_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_out, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("StatArb training complete!")
    for name, score in sorted(all_results.items()):
        print(f"  {name}: {score:.4f}")


if __name__ == '__main__':
    main()
