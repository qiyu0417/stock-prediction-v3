"""并行训练: 多个专家同时训练，充分利用GPU"""
import sys, os, json, time, copy
sys.path.insert(0, 'code/src')
import numpy as np, pandas as pd
import torch, torch.nn.functional as F, joblib
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from ensemble_config import *
from ensemble_models import *
from utils import create_ranking_dataset_vectorized
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, WeightedRankingLoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)

# ============================================================
# 全局数据准备（只做一次）
# ============================================================
def prepare_shared_data():
    """特征工程一次，保存到磁盘供所有进程共享"""
    cache_path = os.path.join(OUTPUT_DIR, '_preprocessed.joblib')
    if os.path.exists(cache_path):
        print(f"加载缓存数据: {cache_path}")
        data = joblib.load(cache_path)
        return data['features'], data['stockid2idx'], data['num_stocks'], data['input_dim']

    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str)
    full_df['日期'] = pd.to_datetime(full_df['日期'], format='mixed')

    all_stocks = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_stocks)}
    num_stocks = len(stockid2idx)

    fe = feature_engineer_func_map[FEATURE_NUM]
    fc = feature_cloums_map[FEATURE_NUM]

    df = full_df.sort_values(['股票代码','日期']).reset_index(drop=True)
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQUENCE_LENGTH + 10]

    print(f"特征工程 ({len(groups)} stocks)...")
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        plist = list(tqdm(pool.imap(fe, groups), total=len(groups), desc='FE'))

    p = pd.concat(plist).reset_index(drop=True)
    p['instrument'] = p['股票代码'].map(stockid2idx)
    p = p.dropna(subset=['instrument'])
    p['instrument'] = p['instrument'].astype(np.int64)
    p = _build_label_and_clean(p, drop_small_open=True)
    p[fc] = p[fc].replace([np.inf, -np.inf], np.nan).dropna(subset=fc)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    p[fc] = scaler.fit_transform(p[fc])

    # 创建排序数据集
    print("创建排序数据集...")
    seqs, tgts, rels, stks = create_ranking_dataset_vectorized(p, fc, SEQUENCE_LENGTH)

    # 从实际数据推断input_dim
    input_dim = seqs[0].shape[-1] if len(seqs) > 0 else len(fc)

    data = {
        'features': fc,
        'stockid2idx': stockid2idx,
        'num_stocks': num_stocks,
        'input_dim': input_dim,
        'sequences': seqs,
        'targets': tgts,
        'relevance': rels,
        'stock_indices': stks,
    }

    joblib.dump(data, cache_path)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    print(f"数据已缓存: {len(seqs)} 样本")
    return fc, stockid2idx, num_stocks, input_dim

# ============================================================
# 单专家训练（在子进程中运行）
# ============================================================
def train_one_expert_worker(args):
    """子进程训练单个专家"""
    expert_cfg, input_dim, num_stocks, gpu_id = args

    # 每个子进程独立加载数据
    cache_path = os.path.join(OUTPUT_DIR, '_preprocessed.joblib')
    data = joblib.load(cache_path)
    seqs, tgts, rels, stks = data['sequences'], data['targets'], data['relevance'], data['stock_indices']

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    set_seed(RANDOM_SEED)

    # 创建模型
    t = expert_cfg.get('type', 'transformer')
    if t == 'transformer':
        model = StockTransformerExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'month_seasonal':
        model = MonthSeasonalExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'aggressive':
        model = AggressiveExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'brownian':
        model = BrownianNoiseExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'statarb':
        model = StatArbRegressionExpert(input_dim, expert_cfg, num_stocks)
    elif t == 'conv':
        model = ConvStockExpert(input_dim, expert_cfg, num_stocks)
    else:
        return None

    model.to(device)
    name = expert_cfg['name']
    is_brownian = (t == 'brownian')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 数据加载器
    ds = RankingDataset(seqs, tgts, rels, stks)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    criterion = WeightedRankingLoss(
        k=5, temperature=1.0, weight_factor=TOP5_WEIGHT,
        pairwise_weight=PAIRWISE_WEIGHT, base_weight=BASE_WEIGHT
    )

    lr = LEARNING_RATE * (3.0 if t == 'aggressive' else 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=NUM_EPOCHS//3, T_mult=2, eta_min=lr*0.01
    )

    print(f"[{name}] 开始训练 | 参数:{n_params:,} | GPU:{gpu_id}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0; steps = 0
        noise_progress = epoch / NUM_EPOCHS if is_brownian else 0.5

        for batch in dl:
            seq = batch['sequences'].to(device)
            masks = batch['masks'].to(device)
            rel = batch['relevance'].to(device)

            optimizer.zero_grad()

            if is_brownian:
                scores = model(seq, epoch_progress=noise_progress, add_noise=True)
            else:
                scores = model(seq)

            masked = scores * masks + (1 - masks) * (-1e9)
            loss = None
            B = seq.size(0)
            for i in range(B):
                vi = masks[i].nonzero().squeeze()
                if vi.numel() <= 1: continue
                if vi.dim() == 0: vi = vi.unsqueeze(0)
                l = criterion(masked[i][vi].unsqueeze(0), rel[i][vi].float().unsqueeze(0))
                loss = loss + l if loss is not None else l

            if loss is not None:
                (loss / B).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                train_loss += loss.item() / B
            steps += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"[{name}] E{epoch+1}/{NUM_EPOCHS} Loss:{train_loss/max(steps,1):.4f}")

    # 保存
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'expert_{name}.pth'))
    print(f"[{name}] 完成! Loss:{train_loss/max(steps,1):.4f}")
    return name

# ============================================================
# 主流程
# ============================================================
def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出: {OUTPUT_DIR}")

    # 1. 准备共享数据
    print("\n=== Step 1: 数据准备 ===")
    features, stockid2idx, num_stocks, input_dim = prepare_shared_data()

    # 2. 选择要训练的专家（跳过已完成的）
    experts_to_train = []
    for cfg in EXPERT_CONFIGS:
        name = cfg['name']
        path = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
        if not os.path.exists(path):
            experts_to_train.append(cfg)
        else:
            print(f"  跳过 {name} (已存在)")

    if not experts_to_train:
        print("所有专家已训练完成!")
    else:
        print(f"\n=== Step 2: 并行训练 {len(experts_to_train)} 个专家 ===")

        # 每个专家一个进程（GPU上并行）
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(f"可用GPU: {n_gpus}")

        # 最多同时跑 min(8, len(experts_to_train)) 个专家
        max_workers = min(8, len(experts_to_train))
        print(f"并行数: {max_workers}")

        # 准备参数
        worker_args = []
        for i, cfg in enumerate(experts_to_train):
            worker_args.append((cfg, input_dim, num_stocks, i % n_gpus))

        # 并行执行
        start = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
            futures = {executor.submit(train_one_expert_worker, arg): arg[0]['name'] for arg in worker_args}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  ✓ {result} 完成 ({len(results)}/{len(experts_to_train)})")
                except Exception as e:
                    print(f"  ✗ {name} 失败: {e}")

        elapsed = time.time() - start
        print(f"\n并行训练完成! 耗时: {elapsed/60:.1f}分钟")

    # 3. 训练元调度器
    print(f"\n=== Step 3: 元调度器 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载缓存数据
    cache_path = os.path.join(OUTPUT_DIR, '_preprocessed.joblib')
    data = joblib.load(cache_path)
    seqs, tgts, rels, stks = data['sequences'], data['targets'], data['relevance'], data['stock_indices']

    # 加载所有专家
    experts = []
    for cfg in EXPERT_CONFIGS:
        name = cfg['name']
        path = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
        if not os.path.exists(path):
            print(f"  跳过 {name}")
            continue
        t = cfg.get('type', 'transformer')
        if t == 'transformer': m = StockTransformerExpert(input_dim, cfg, num_stocks)
        elif t == 'month_seasonal': m = MonthSeasonalExpert(input_dim, cfg, num_stocks)
        elif t == 'aggressive': m = AggressiveExpert(input_dim, cfg, num_stocks)
        elif t == 'brownian': m = BrownianNoiseExpert(input_dim, cfg, num_stocks)
        elif t == 'statarb': m = StatArbRegressionExpert(input_dim, cfg, num_stocks)
        elif t == 'conv': m = ConvStockExpert(input_dim, cfg, num_stocks)
        else: continue
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device).eval()
        experts.append(m)
        print(f"  {name} ✓")
    print(f"共 {len(experts)} 个专家")

    # 训练Meta
    ds = RankingDataset(seqs, tgts, rels, stks)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    print("收集专家预测...")
    all_scores, all_tgt, all_mask = [], [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc="推理"):
            seq = batch['sequences'].to(device)
            batch_scores = []
            for e in experts:
                e.eval()
                if hasattr(e, 'predict_with_mc_dropout'):
                    batch_scores.append(e.predict_with_mc_dropout(seq, num_samples=5))
                else:
                    batch_scores.append(e(seq))
            all_scores.append(torch.stack(batch_scores, dim=-1))
            all_tgt.append(batch['targets'])
            all_mask.append(batch['masks'])

    meta = MetaAggregator(len(experts), num_stocks, hidden_dim=META_HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(meta.parameters(), lr=META_LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=META_EPOCHS//3, T_mult=2, eta_min=META_LR*0.01)

    print(f"训练元调度器 ({META_EPOCHS} epochs)...")
    for epoch in range(META_EPOCHS):
        meta.train()
        total_loss = 0; nb = 0
        for i in range(len(all_scores)):
            es = all_scores[i].to(device); tg = all_tgt[i].to(device); mk = all_mask[i].to(device)
            opt.zero_grad()
            mf = meta(es) * mk + (1 - mk) * (-1e9)
            loss = None; B = es.size(0)
            for j in range(B):
                vi = mk[j].nonzero().squeeze()
                if vi.numel() <= 1: continue
                if vi.dim() == 0: vi = vi.unsqueeze(0)
                vp = mf[j][vi]; vt = tg[j][vi]
                _, si = torch.sort(vt, descending=True)
                r = torch.zeros_like(vt)
                r[si] = torch.arange(len(vt), 0, -1, device=device, dtype=torch.float32)
                l = F.mse_loss(vp, r)
                loss = loss + l if loss is not None else l
            if loss is not None:
                (loss / B).backward(); opt.step()
                total_loss += loss.item() / B; nb += 1
        sch.step()
        if (epoch + 1) % 5 == 0:
            print(f"  [Meta] E{epoch+1} Loss:{total_loss/max(nb,1):.4f}")

    torch.save(meta.state_dict(), os.path.join(OUTPUT_DIR, 'meta_aggregator.pth'))

    # 保存配置
    final_cfg = {
        'sequence_length': SEQUENCE_LENGTH, 'feature_num': FEATURE_NUM,
        'input_dim': input_dim, 'expert_configs': EXPERT_CONFIGS,
        'num_stocks': num_stocks, 'stockid2idx': stockid2idx,
        'feature_list': features, 'mc_samples': MC_SAMPLES,
    }
    json.dump(final_cfg, open(os.path.join(OUTPUT_DIR, 'ensemble_config.json'), 'w'), indent=2, ensure_ascii=False)

    print(f"\n=== 全部完成! {len(experts)}专家 ===")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
