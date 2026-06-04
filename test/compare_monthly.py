"""
V1 vs V3 月度收益对比 (2026年1月-5月)
- 每月最后5个交易日作为测试期
- 使用测试期前数据预测，计算5日加权收益
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
from collections import Counter

from config_v3 import *
from ensemble_models import (
    StockTransformerExpert, ConvStockExpert, MonthSeasonalExpert, MetaAggregator
)
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from risk_filter import compute_risk_scores, apply_risk_filter

TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
V1_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v1_ensemble')
V3_EXPERT_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v2_ensemble')


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_for_date(df, stockid2idx):
    """预处理到指定日期"""
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    processed = pd.concat([feature_engineer(g) for g in tqdm(groups, desc='  特征工程', leave=False)]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])
    return processed, feature_columns


def build_sequences(data, features, stock_ids, target_date):
    sequences, seq_stock_ids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= target_date)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            sequences.append(hist[features].values.astype(np.float32))
            seq_stock_ids.append(sid)
    return np.asarray(sequences, dtype=np.float32) if sequences else np.array([]), seq_stock_ids


def load_v1_experts(feature_dim, num_stocks, device):
    """加载 V1 专家 + MetaAggregator"""
    with open(os.path.join(V1_DIR, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    expert_cfgs = cfg['expert_configs']

    experts = []
    for ec in expert_cfgs:
        t = ec.get('type', 'transformer')
        if t == 'transformer':
            m = StockTransformerExpert(feature_dim, ec, num_stocks)
        elif t == 'conv':
            m = ConvStockExpert(feature_dim, ec, num_stocks)
        elif t == 'month_seasonal':
            m = MonthSeasonalExpert(feature_dim, ec, num_stocks)
        else:
            continue
        path = os.path.join(V1_DIR, f'expert_{ec["name"]}.pth')
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
            m.to(device)
            experts.append(m)
    meta = MetaAggregator(len(experts), num_stocks, hidden_dim=64)
    meta_path = os.path.join(V1_DIR, 'meta_aggregator.pth')
    if os.path.exists(meta_path):
        meta.load_state_dict(torch.load(meta_path, map_location=device))
    meta.to(device)
    return experts, meta, cfg


def v1_predict(experts, meta, x, device):
    """V1: MetaAggregator 融合"""
    all_scores = []
    for expert in experts:
        expert.train()
        mc = []
        with torch.no_grad():
            for _ in range(20):
                mc.append(expert(x).squeeze(0))
        all_scores.append(torch.stack(mc).mean(dim=0))
    stack = torch.stack(all_scores, dim=-1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        fused = meta(stack).squeeze(0).cpu().numpy()
    return fused


def v3_predict(experts, weights, x, device, seq_ids, risk_scores, market_stress):
    """V3: 加权 + 投票共识 + 风险过滤"""
    NUM_ROUNDS = 5
    MC_SPR = 30
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'

    all_top5 = []
    for r in range(NUM_ROUNDS):
        torch.manual_seed(42 + r * 100)
        np.random.seed(42 + r * 100)
        rnd_scores = []
        for expert in experts:
            expert.train()
            mc = []
            with torch.no_grad():
                for _ in range(MC_SPR):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x.size(1) <= chunk_size:
                            s = expert(x).squeeze(0)
                        else:
                            cs = []
                            for start in range(0, x.size(1), chunk_size):
                                end = min(start + chunk_size, x.size(1))
                                cs.append(expert(x[:, start:end].contiguous()).squeeze(0))
                            s = torch.cat(cs, dim=0)
                    mc.append(s)
            rnd_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())
        fused = np.zeros(len(rnd_scores[0]))
        for w, sc in zip(weights, rnd_scores):
            fused += w * sc
        sel, _ = apply_risk_filter(fused, seq_ids, risk_scores, market_stress,
                                   max_risk_score=80, min_positions=1, max_positions=5)
        all_top5.extend(sel)

    vc = Counter(all_top5)
    consensus = [s for s, c in vc.most_common() if c >= 3]
    if len(consensus) < 1:
        consensus = [s for s, _ in vc.most_common(3)]
    return consensus


def calc_5d_return(stock_ids, weights, test_data):
    """计算加权5日收益"""
    total_ret = 0
    for sid, w in zip(stock_ids, weights):
        stock_test = test_data[test_data['股票代码'] == sid].sort_values('日期')
        if len(stock_test) >= 2:
            start_open = stock_test.iloc[0]['开盘']
            end_open = stock_test.iloc[-1]['开盘']
            ret = (end_open - start_open) / start_open
            total_ret += w * ret
    return total_ret


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"设备: {device}")

    full_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    full_df['year_month'] = full_df['日期'].dt.to_period('M')

    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    months = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05']
    test_dates_map = {}
    for ym in months:
        mdata = full_df[full_df['year_month'] == ym]
        test_dates_map[ym] = sorted(mdata['日期'].unique())[-5:]

    # 预加载 V1 模型 (需要先知道特征维度)
    # 先用完整数据做一次预处理获取特征维度
    temp_df = full_df[full_df['日期'] <= full_df['日期'].max()].copy()
    temp_proc, feats = preprocess_for_date(temp_df, stockid2idx)
    feature_dim = len(feats)
    print(f"特征维度: {feature_dim}")

    # 加载模型
    print("\n加载 V1 模型...")
    v1_experts, v1_meta, v1_cfg = load_v1_experts(feature_dim, num_stocks, device)
    print(f"  V1: {len(v1_experts)} experts + MetaAggregator")

    print("加载 V3 模型...")
    v3_models = []
    v3_names = ['balanced_v2', 'deep_v2', 'conv_multiscale', 'conv_deep']
    v3_weights_raw = [0.1855, 0.1215, 0.1113, 0.0804]
    v3_weights = [w / sum(v3_weights_raw) for w in v3_weights_raw]

    for name in v3_names:
        path = os.path.join(V3_EXPERT_DIR, f'expert_{name}.pth')
        if name.startswith('conv'):
            cfg = {'name': name, 'type': 'conv', 'hidden_channels': 256 if 'multi' in name else 384,
                   'nhead': 4, 'dropout': 0.12 if 'multi' in name else 0.15,
                   'mc_dropout_rate': 0.1 if 'multi' in name else 0.12, 'sd_prob': 0.9 if 'multi' in name else 0.85}
            m = ConvStockExpert(feature_dim, cfg, num_stocks)
        else:
            cfg = {'name': name, 'type': 'transformer',
                   'd_model': 256 if name == 'balanced_v2' else 192, 'nhead': 4,
                   'num_layers': 6 if name == 'balanced_v2' else 8,
                   'dim_feedforward': 512 if name == 'balanced_v2' else 384,
                   'dropout': 0.1, 'mc_dropout_rate': 0.1 if name == 'balanced_v2' else 0.12,
                   'sd_prob': 0.9 if name == 'balanced_v2' else 0.85}
            m = StockTransformerExpert(feature_dim, cfg, num_stocks)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        v3_models.append(m)
    print(f"  V3: {len(v3_models)} experts + 投票共识")

    # 加载 scaler
    scaler_path = os.path.join(V3_EXPERT_DIR, 'scaler.pkl')
    scaler = joblib.load(scaler_path)

    results = []

    for ym in months:
        print(f"\n{'='*50}")
        print(f"{ym}")
        print(f"{'='*50}")

        test_dates = test_dates_map[ym]
        cutoff_date = test_dates[0] - pd.Timedelta(days=1)
        print(f"  截止日期: {cutoff_date.date()}, 测试期: {test_dates[0].date()} ~ {test_dates[-1].date()}")

        # 准备数据
        train_df = full_df[full_df['日期'] <= cutoff_date].copy()
        processed, features = preprocess_for_date(train_df, stockid2idx)
        processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 风险评分
        risk_raw, stress = compute_risk_scores(
            processed, features, stock_ids, stock_ids,
            train_df['日期'].max()
        )

        # 标准化
        common = [c for c in scaler.feature_names_in_ if c in processed.columns]
        processed[common] = scaler.transform(processed[common])
        features = common

        # 构建序列
        pred_date = cutoff_date
        seq_np, seq_ids = build_sequences(processed, features, stock_ids, pred_date)
        print(f"  可用股票: {len(seq_ids)}")

        risk_scores = {sid: risk_raw.get(sid, 50) for sid in seq_ids}
        x = torch.from_numpy(seq_np).unsqueeze(0).to(device)

        # 测试集数据 (用于计算收益)
        test_data = full_df[full_df['日期'].isin(test_dates)].copy()

        # --- V1 预测 ---
        print("  V1 预测中...")
        v1_fused = v1_predict(v1_experts, v1_meta, x, device)
        v1_order = np.argsort(v1_fused)[::-1]
        v1_top5 = [seq_ids[i] for i in v1_order[:5]]
        v1_ret = calc_5d_return(v1_top5, [0.2]*5, test_data)

        # --- V3 预测 ---
        print("  V3 预测中...")
        v3_top = v3_predict(v3_models, v3_weights, x, device, seq_ids, risk_scores, stress)
        if len(v3_top) > 5:
            v3_top = v3_top[:5]
        v3_w = [1.0/len(v3_top)] * len(v3_top) if v3_top else []
        v3_ret = calc_5d_return(v3_top, v3_w, test_data) if v3_top else 0

        results.append({
            'month': ym,
            'cutoff': str(cutoff_date.date()),
            'v1_top5': v1_top5,
            'v1_return': v1_ret,
            'v3_top5': v3_top,
            'v3_return': v3_ret,
            'v3_positions': len(v3_top),
        })

        print(f"  V1 Top5: {v1_top5} → {v1_ret:+.4%}")
        print(f"  V3 Top{len(v3_top)}: {v3_top} → {v3_ret:+.4%}")

    # 汇总
    print(f"\n{'='*70}")
    print("月度收益对比: V1 vs V3")
    print(f"{'='*70}")
    print(f"{'月份':<10} {'V1 Top5':<30} {'V1收益':>8} {'V3 TopN':<30} {'V3收益':>8} {'仓位':>5}")
    print("-" * 85)
    v1_total = 0
    v3_total = 0
    for r in results:
        v1_str = ','.join(r['v1_top5'][:3]) + '..'
        v3_str = ','.join(r['v3_top5'][:3]) + '..'
        print(f"{r['month']:<10} {v1_str:<30} {r['v1_return']:>+7.2%} {v3_str:<30} {r['v3_return']:>+7.2%} {r['v3_positions']:>4}")
        v1_total += r['v1_return']
        v3_total += r['v3_return']
    print("-" * 85)
    print(f"{'累计':<10} {'':<30} {v1_total:>+7.2%} {'':<30} {v3_total:>+7.2%}")
    print(f"{'均值':<10} {'':<30} {v1_total/5:>+7.2%} {'':<30} {v3_total/5:>+7.2%}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
