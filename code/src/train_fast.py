"""
快速训练: 月份季节性专家 + 元调度器
复用已完成5个Transformer专家
"""
import os, sys, json, random
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
    StockTransformerExpert, MonthSeasonalExpert, MetaAggregator
)
from utils import create_ranking_dataset_vectorized
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, WeightedRankingLoss,
    calculate_ranking_metrics, RankingDataset, collate_fn
)

# 模型工厂
def create_model(cfg, input_dim, num_stocks):
    t = cfg.get('type', 'transformer')
    if t == 'transformer':
        return StockTransformerExpert(input_dim, cfg, num_stocks)
    elif t == 'month_seasonal':
        return MonthSeasonalExpert(input_dim, cfg, num_stocks)
    raise ValueError(f"Unknown: {t}")

# 数据分割（和主训练一致）
def create_rolling_window_splits(df, seq_len):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df['year_month'] = df['日期'].dt.to_period('M')
    months = sorted(df['year_month'].unique())
    windows = []
    for i in range(len(months) - WINDOW_SIZE_MONTHS + 1):
        wm = months[i:i+WINDOW_SIZE_MONTHS]
        rng = random.Random(RANDOM_SPLIT_SEED + i)
        indices = list(range(WINDOW_SIZE_MONTHS))
        rng.shuffle(indices)
        val_idx = set(indices[:VAL_MONTHS])
        train_m = [m for j,m in enumerate(wm) if j not in val_idx]
        val_m = [m for j,m in enumerate(wm) if j in val_idx]

        train_df = df[df['year_month'].isin(train_m)].copy()
        val_df = df[df['year_month'].isin(val_m)].copy()

        if len(train_df) > 0:
            train_min = train_df['日期'].min()
            ctx_start = train_min - pd.tseries.offsets.BDay(seq_len)
            train_df = pd.concat([df[(df['日期']>=ctx_start)&(df['日期']<train_min)].copy(), train_df])
        if len(val_df) > 0:
            val_min = val_df['日期'].min()
            ctx_start = val_min - pd.tseries.offsets.BDay(seq_len)
            val_df = pd.concat([df[(df['日期']>=ctx_start)&(df['日期']<val_min)].copy(), val_df])

        windows.append({
            'train_df': train_df, 'val_df': val_df,
            'train_months': [str(m) for m in train_m],
            'val_months': [str(m) for m in val_m],
            'val_start': str(val_df['日期'].min().date()) if len(val_df)>0 else None
        })
    return windows

def preprocess_window(train_df, val_df, stockid2idx, feature_num, seq_len):
    fe = feature_engineer_func_map[feature_num]
    fc = feature_cloums_map[feature_num]

    def process(df, desc):
        df = df.sort_values(['股票代码','日期']).reset_index(drop=True)
        groups = [g.reset_index(drop=True) for _,g in df.groupby('股票代码',sort=False) if len(g)>=seq_len+10]
        if not groups: return None, None
        with mp.Pool(min(8, mp.cpu_count())) as pool:
            plist = list(tqdm(pool.imap(fe, groups), total=len(groups), desc=desc, leave=False))
        p = pd.concat(plist).reset_index(drop=True)
        p['instrument'] = p['股票代码'].map(stockid2idx)
        p = p.dropna(subset=['instrument'])
        p['instrument'] = p['instrument'].astype(np.int64)
        p = _build_label_and_clean(p, drop_small_open=True)
        return p, fc

    train_p, features = process(train_df, "训练特征")
    if train_p is None: return None,None,None,None
    val_p, _ = process(val_df, "验证特征") if val_df is not None and len(val_df)>0 else (None, None)

    for d in [train_p, val_p]:
        if d is not None:
            d[features] = d[features].replace([np.inf,-np.inf], np.nan).dropna(subset=features)

    scaler = StandardScaler()
    train_p[features] = scaler.fit_transform(train_p[features])
    if val_p is not None:
        val_p[features] = scaler.transform(val_p[features])
    return train_p, val_p, features, scaler

def train_one_expert(model, cfg, train_data, val_data, features, seq_len, device, name):
    is_adversarial = (cfg.get('type') == 'adversarial')

    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(train_data, features, seq_len)
    val_seq, val_tgt, val_rel, val_stk = None,None,None,None
    if val_data is not None:
        val_seq, val_tgt, val_rel, val_stk = create_ranking_dataset_vectorized(val_data, features, seq_len)

    if len(train_seq) == 0: return None

    train_ds = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    val_dl = None
    if val_seq and len(val_seq) > 0:
        val_ds = RankingDataset(val_seq, val_tgt, val_rel, val_stk)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    criterion = WeightedRankingLoss(k=5, temperature=1.0, weight_factor=TOP5_WEIGHT, pairwise_weight=PAIRWISE_WEIGHT, base_weight=BASE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=LEARNING_RATE*0.005)

    best_score = -float('inf')
    best_state = None
    patience = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        steps = 0

        for batch in train_dl:
            seq = batch['sequences'].to(device)
            masks = batch['masks'].to(device)
            rel = batch['relevance'].to(device)

            optimizer.zero_grad()
            scores = model(seq)
            masked = scores * masks + (1-masks)*(-1e9)

            loss = None
            B = seq.size(0)
            for i in range(B):
                vi = masks[i].nonzero().squeeze()
                if vi.numel() <= 1: continue
                if vi.dim() == 0: vi = vi.unsqueeze(0)
                vp = masked[i][vi]
                vr = rel[i][vi].float()
                l = criterion(vp.unsqueeze(0), vr.unsqueeze(0))
                loss = loss + l if loss is not None else l

            if loss is not None:
                (loss/B).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                train_loss += loss.item()/B
            steps += 1

        scheduler.step()
        train_loss /= max(steps, 1)

        val_score = None
        if val_dl:
            model.eval()
            vs = {}
            vn = 0
            with torch.no_grad():
                for batch in val_dl:
                    seq = batch['sequences'].to(device)
                    tgt = batch['targets'].to(device)
                    masks = batch['masks'].to(device)
                    scores = model(seq)
                    masked = scores * masks + (1-masks)*(-1e9)
                    m = calculate_ranking_metrics(masked, tgt*masks, masks, k=5)
                    for k,v in m.items(): vs[k] = vs.get(k,0)+v
                    vn += 1
            if vn > 0:
                for k in vs: vs[k] /= vn
            val_score = vs.get('final_score', 0)

            if val_score > best_score + EARLY_STOPPING_MIN_DELTA:
                best_score = val_score
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

        if (epoch+1) % 5 == 0:
            print(f"  [{name}] E{epoch+1:2d} Loss:{train_loss:.4f} Val:{val_score:.4f}" if val_score is not None else f"  [{name}] E{epoch+1:2d}")

        if val_dl and patience >= EARLY_STOPPING_PATIENCE:
            print(f"  [{name}] 早停! 最佳:{best_score:.4f}")
            break

    if best_state: model.load_state_dict(best_state)
    return best_score

def train_meta(experts, meta, train_data, features, seq_len, device):
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(train_data, features, seq_len)
    if len(train_seq) == 0: return

    train_ds = RankingDataset(train_seq, train_tgt, train_rel, train_stk)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # 收集专家预测
    print("收集专家预测...")
    all_scores, all_tgt, all_mask = [], [], []
    with torch.no_grad():
        for batch in tqdm(train_dl, desc="专家推理"):
            seq = batch['sequences'].to(device)
            batch_scores = []
            for e in experts:
                e.eval()
                batch_scores.append(e.predict_with_mc_dropout(seq, num_samples=5))
            all_scores.append(torch.stack(batch_scores, dim=-1))
            all_tgt.append(batch['targets'])
            all_mask.append(batch['masks'])

    print(f"训练元调度器 ({META_EPOCHS} epochs)...")
    opt = torch.optim.Adam(meta.parameters(), lr=META_LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=META_EPOCHS//3, T_mult=2, eta_min=META_LR*0.01)

    for epoch in range(META_EPOCHS):
        meta.train()
        total_loss = 0
        nb = 0
        for i in range(len(all_scores)):
            es = all_scores[i].to(device)
            tg = all_tgt[i].to(device)
            mk = all_mask[i].to(device)

            opt.zero_grad()
            final = meta(es)
            mf = final * mk + (1-mk)*(-1e9)

            loss = None
            B = es.size(0)
            for j in range(B):
                vi = mk[j].nonzero().squeeze()
                if vi.numel() <= 1: continue
                if vi.dim()==0: vi=vi.unsqueeze(0)
                vp = mf[j][vi]
                vt = tg[j][vi]
                _, si = torch.sort(vt, descending=True)
                r = torch.zeros_like(vt)
                r[si] = torch.arange(len(vt),0,-1,device=device,dtype=torch.float32)
                l = F.mse_loss(vp, r)
                loss = loss+l if loss is not None else l

            if loss is not None:
                (loss/B).backward()
                opt.step()
                total_loss += loss.item()/B
                nb += 1
        sch.step()
        if nb > 0 and (epoch+1)%5==0:
            print(f"  [Meta] E{epoch+1} Loss:{total_loss/max(nb,1):.4f}")

def main():
    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"输出: {OUTPUT_DIR}")

    # 加载数据
    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码':str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    all_stocks = sorted(full_df['股票代码'].unique())
    stockid2idx = {s:i for i,s in enumerate(all_stocks)}
    num_stocks = len(stockid2idx)
    print(f"股票: {num_stocks}, 行数: {len(full_df)}")

    # 滚动窗口
    windows = create_rolling_window_splits(full_df, SEQUENCE_LENGTH)
    print(f"窗口: {len(windows)}")

    # 预处理所有窗口
    print("预处理窗口...")
    wdata_list = []
    for w in windows:
        tp, vp, feats, scaler = preprocess_window(w['train_df'], w['val_df'], stockid2idx, FEATURE_NUM, SEQUENCE_LENGTH)
        if tp is not None and len(tp) > 100:
            wdata_list.append({**w, 'train':tp, 'val':vp, 'features':feats, 'scaler':scaler})
    feature_list = wdata_list[0]['features']
    input_dim = len(feature_list)
    print(f"有效窗口: {len(wdata_list)}, 特征: {input_dim}")

    # 月份专家配置
    month_cfg = [c for c in EXPERT_CONFIGS if c.get('type')=='month_seasonal'][0]
    print(f"\n=== 训练月份季节性专家 ===")

    month_model = MonthSeasonalExpert(input_dim, month_cfg, num_stocks).to(device)
    print(f"参数量: {sum(p.numel() for p in month_model.parameters() if p.requires_grad):,}")

    best_month_score = -float('inf')
    best_month_state = None
    for wd in wdata_list:
        score = train_one_expert(month_model, month_cfg, wd['train'], wd['val'], feature_list, SEQUENCE_LENGTH, device, 'month_seasonal')
        if score is not None and score > best_month_score:
            best_month_score = score
            best_month_state = {k:v.cpu().clone() for k,v in month_model.state_dict().items()}

    if best_month_state: month_model.load_state_dict(best_month_state)
    torch.save(month_model.state_dict(), os.path.join(OUTPUT_DIR, 'expert_month_seasonal.pth'))
    print(f"月份专家 最佳: {best_month_score:.4f}, 已保存")

    # 加载5个Transformer专家
    print(f"\n=== 加载已完成专家 ===")
    experts = []
    for cfg in EXPERT_CONFIGS:
        name = cfg['name']
        path = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
        if os.path.exists(path):
            m = create_model(cfg, input_dim, num_stocks).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            experts.append(m)
            print(f"  已加载: {name}")
        elif cfg['type'] == 'month_seasonal':
            experts.append(month_model)
            print(f"  已加载: {name} (刚训练)")

    print(f"共 {len(experts)} 个专家")

    # 训练元调度器
    print(f"\n=== 训练元调度器 ===")
    meta = MetaAggregator(len(experts), num_stocks, hidden_dim=META_HIDDEN_DIM).to(device)
    # 用最后几个窗口的数据
    meta_data = pd.concat([w['train'] for w in wdata_list[-3:]])
    train_meta(experts, meta, meta_data, feature_list, SEQUENCE_LENGTH, device)
    torch.save(meta.state_dict(), os.path.join(OUTPUT_DIR, 'meta_aggregator.pth'))

    # 保存配置
    final_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'feature_num': FEATURE_NUM,
        'input_dim': input_dim,
        'expert_configs': EXPERT_CONFIGS,
        'num_stocks': num_stocks,
        'stockid2idx': stockid2idx,
        'feature_list': feature_list,
        'mc_samples': MC_SAMPLES,
    }
    json.dump(final_config, open(os.path.join(OUTPUT_DIR, 'ensemble_config.json'), 'w'), indent=2, ensure_ascii=False)
    # 保存最后一个scaler
    joblib.dump(wdata_list[-1]['scaler'], os.path.join(OUTPUT_DIR, 'scaler.pkl'))

    print(f"\n=== 全部完成! ===")
    print(f"专家数: {len(experts)}")
    print(f"元调度器: 已训练")
    print(f"输出: {OUTPUT_DIR}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
