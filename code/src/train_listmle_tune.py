"""
ListMLE 调参训练: k=3/10, temperature=0.5/1.0
用法: python train_listmle_tune.py --k 10 --temperature 0.5
"""
import os, sys, json, gc, argparse
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_stock_emb_8 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, ListMLELoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)
from train_stock_emb_8_loss import preprocess_with_winsor, _train_step, _train_chunked

# Modify ListMLELoss to accept temperature
class ListMLETuned(nn.Module):
    """ListMLE with temperature"""
    def __init__(self, k=5, temperature=1.0):
        super().__init__()
        self.k = k
        self.temperature = temperature

    def forward(self, y_pred, y_true):
        batch, n = y_pred.shape
        k = min(self.k, n)
        scaled_pred = y_pred / self.temperature
        _, idx = torch.sort(y_true, dim=1, descending=True)
        losses = []
        for b in range(batch):
            for pos in range(k):
                remaining = idx[b, pos:]
                log_denom = torch.logsumexp(scaled_pred[b, remaining], dim=0)
                losses.append(log_denom - scaled_pred[b, idx[b, pos]])
        return torch.stack(losses).mean()


def train_expert(model, exp_cfg, dataset, device, expert_name, criterion):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pin_memory = device.type == 'cuda'
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=pin_memory)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_stocks', type=int, default=0,
                        help='Only use top N stocks by average volume (0=all)')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tag = f'listmle_k{args.k}_t{args.temperature}'
    stock_filter_str = f'_top{args.top_stocks}' if args.top_stocks > 0 else ''
    print(f"ListMLE Tuning: k={args.k}, temp={args.temperature}, top_stocks={args.top_stocks or 'all'} | Device: {device}")
    set_seed(42)

    output_dir = os.path.join('./model', f'stock_emb_8_{tag}{stock_filter_str}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n加载数据...")
    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    print(f"  总数据: {len(full_df)} 行")

    if args.top_stocks > 0:
        avg_vol = full_df.groupby('股票代码')['成交量'].mean().sort_values(ascending=False)
        top_ids = set(avg_vol.head(args.top_stocks).index)
        full_df = full_df[full_df['股票代码'].isin(top_ids)].copy()
        print(f"  筛选后: {len(full_df)} 行, {len(top_ids)} 只股票")

    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    print("\n预处理...")
    processed, features, scaler, winsor_bounds = preprocess_with_winsor(full_df, stockid2idx)
    n_feats = len(features)
    print(f"  特征维度: {n_feats}")
    del full_df; gc.collect()

    print("\n构建排名数据集...")
    from utils import create_ranking_dataset_vectorized
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, features, SEQUENCE_LENGTH)
    print(f"  训练天数: {len(train_seq)}")
    dataset = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    del processed, train_tgt, train_rel, train_stk; gc.collect()

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    with open(os.path.join(output_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f)

    criterion = ListMLETuned(k=args.k, temperature=args.temperature)
    all_results = {}

    for exp_cfg in EXPERT_CONFIGS:
        name = exp_cfg['name']
        model_path = os.path.join(output_dir, f'expert_{name}.pth')
        if os.path.exists(model_path):
            print(f"\n  SKIP {name}: already trained")
            all_results[name] = 0.0
            continue

        print(f"\n{'='*50}")
        print(f"Training: {name} (k={args.k}, T={args.temperature})")

        if exp_cfg['type'] == 'transformer':
            model = StockTransformerExpert(n_feats, exp_cfg, num_stocks)
        else:
            model = ConvStockExpert(n_feats, exp_cfg, num_stocks)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        model.to(device)

        best_score = train_expert(model, exp_cfg, dataset, device, name, criterion)
        all_results[name] = best_score

        torch.save(model.state_dict(), model_path)
        print(f"  Saved: expert_{name}.pth (score={best_score:.4f})")
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    config_out = {
        'feature_dim': n_feats,
        'num_stocks': num_stocks,
        'expert_configs': EXPERT_CONFIGS,
        'expert_results': {k: float(v) for k, v in all_results.items()},
        'features': FEATURE_NUM,
        'loss_type': f'listmle_k{args.k}_t{args.temperature}',
    }
    with open(os.path.join(output_dir, 'ensemble_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_out, f, ensure_ascii=False, indent=2)

    print(f"\nDone! k={args.k}, T={args.temperature}")
    for name, score in sorted(all_results.items()):
        print(f"  {name}: {score:.4f}")


if __name__ == '__main__':
    main()
