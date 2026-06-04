"""
V1 vs V3 官方测试集对比 (test.csv: 2026-05-28 ~ 2026-06-03)
- 使用 train.csv 中 ≤2026-05-27 的数据做预测
- 评分逻辑与 score_self.py 完全一致
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
TEST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'test.csv')
V1_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v1_ensemble')
V3_EXPERT_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v2_ensemble')

CUTOFF_DATE = '2026-05-27'
TEST_DATES = ['2026-05-28', '2026-05-29', '2026-06-01', '2026-06-02', '2026-06-03']


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_for_date(df, stockid2idx):
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


def v3_predict(experts, weights, x, device, seq_ids, risk_scores, market_stress,
               max_risk_score=80, min_positions=1, max_positions=5):
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
                                   max_risk_score=max_risk_score, min_positions=min_positions,
                                   max_positions=max_positions)
        all_top5.extend(sel)

    vc = Counter(all_top5)
    consensus = [s for s, c in vc.most_common() if c >= 3]
    if len(consensus) < 1:
        consensus = [s for s, _ in vc.most_common(3)]
    return consensus


def calc_official_score(stock_ids, weights, test_data):
    """与 score_self.py 完全一致的评分逻辑"""
    filtered = test_data[test_data['股票代码'].isin(stock_ids)]
    if filtered.empty:
        return 0.0

    total = 0.0
    for sid, w in zip(stock_ids, weights):
        stock_test = filtered[filtered['股票代码'] == sid].sort_values('日期').tail(5)
        if len(stock_test) >= 2:
            start_open = stock_test.iloc[0]['开盘']
            end_open = stock_test.iloc[-1]['开盘']
            ret = (end_open - start_open) / start_open
            total += w * ret
    return total


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"设备: {device}")

    # 加载训练数据
    full_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    # 加载官方测试数据
    test_df = pd.read_csv(TEST_PATH, dtype={'股票代码': str})
    test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
    test_df['日期'] = pd.to_datetime(test_df['日期'])
    test_stocks = set(test_df['股票代码'].unique())
    print(f"测试集: {len(test_df)} 行, {len(test_stocks)} 只股票, "
          f"{test_df['日期'].min().date()} ~ {test_df['日期'].max().date()}")

    # 训练集截止到 CUTOFF_DATE
    train_df = full_df[full_df['日期'] <= CUTOFF_DATE].copy()
    print(f"训练数据: {len(train_df)} 行, 截止 {CUTOFF_DATE}")

    # 只用测试集中出现的股票 (但需要足够的训练历史)
    stock_ids = sorted(test_stocks)
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # 预处理获取特征维度
    print("\n预处理...")
    processed, features = preprocess_for_date(train_df, stockid2idx)
    feature_dim = len(features)
    print(f"特征维度: {feature_dim}")

    # 风险评分
    risk_raw, stress = compute_risk_scores(
        processed, features, stock_ids, stock_ids,
        train_df['日期'].max()
    )

    # 标准化
    scaler_path = os.path.join(V3_EXPERT_DIR, 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    common = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[common] = scaler.transform(processed[common])
    features_std = common

    # 构建序列
    pred_date = pd.to_datetime(CUTOFF_DATE)
    seq_np, seq_ids = build_sequences(processed, features_std, stock_ids, pred_date)
    print(f"可用股票: {len(seq_ids)} / 测试集股票: {len(test_stocks)}")

    risk_scores = {sid: risk_raw.get(sid, 50) for sid in seq_ids}
    x = torch.from_numpy(seq_np).unsqueeze(0).to(device)

    # --- 加载 V1 ---
    print("\n加载 V1 模型...")
    v1_experts, v1_meta, _ = load_v1_experts(feature_dim, num_stocks, device)
    print(f"  V1: {len(v1_experts)} experts + MetaAggregator")

    # --- 加载 V3 ---
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

    # --- V1 预测 ---
    print("\n" + "=" * 50)
    print("V1 预测中...")
    v1_fused = v1_predict(v1_experts, v1_meta, x, device)
    v1_order = np.argsort(v1_fused)[::-1]
    v1_top5 = [seq_ids[i] for i in v1_order[:5]]
    v1_weights = [0.2] * 5
    v1_score = calc_official_score(v1_top5, v1_weights, test_df)

    print(f"  V1 Top5: {v1_top5}")
    # 打印每只股票的详细收益
    v1_details = test_df[test_df['股票代码'].isin(v1_top5)]
    for sid in v1_top5:
        s = v1_details[v1_details['股票代码'] == sid].sort_values('日期')
        if len(s) >= 2:
            ret = (s.iloc[-1]['开盘'] - s.iloc[0]['开盘']) / s.iloc[0]['开盘']
            print(f"    {sid}: {ret:+.4%}")

    # --- V3 预测 (多组风险参数) ---
    risk_configs = [
        ('保守 (max=80, min=1)', 80, 1, 5),
        ('适中 (max=85, min=3)', 85, 3, 5),
        ('宽松 (max=90, min=3)', 90, 3, 5),
        ('无过滤 (直接Top5)', 100, 5, 5),
    ]

    print("\n" + "=" * 50)
    print("V3 预测 (多组风险参数)...")
    v3_results = []
    for label, max_risk, min_pos, max_pos in risk_configs:
        v3_top = v3_predict(v3_models, v3_weights, x, device, seq_ids, risk_scores, stress,
                            max_risk_score=max_risk, min_positions=min_pos, max_positions=max_pos)
        if len(v3_top) > 5:
            v3_top = v3_top[:5]
        v3_w = [1.0 / len(v3_top)] * len(v3_top) if v3_top else []
        v3_s = calc_official_score(v3_top, v3_w, test_df) if v3_top else 0
        v3_results.append((label, v3_top, v3_w, v3_s))

        v3_details = test_df[test_df['股票代码'].isin(v3_top)]
        print(f"  {label}: Top{len(v3_top)} {v3_top} → {v3_s:+.4%}")
        for sid in v3_top:
            s = v3_details[v3_details['股票代码'] == sid].sort_values('日期')
            if len(s) >= 2:
                ret = (s.iloc[-1]['开盘'] - s.iloc[0]['开盘']) / s.iloc[0]['开盘']
                print(f"    {sid}: {ret:+.4%}")

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("官方测试集对比 (2026-05-28 ~ 2026-06-03)")
    print("=" * 60)
    print(f"  {'':<10} {'股票':<35} {'仓位':<6} {'得分'}")
    print(f"  {'V1':<10} {','.join(v1_top5):<35} {'5':<6} {v1_score:+.4%}")
    for label, stocks, weights, score in v3_results:
        n = len(stocks)
        s_str = ','.join(stocks) if stocks else '(无)'
        print(f"  V3-{label:<6} {s_str:<35} {n:<6} {score:+.4%}")

    # 保存各版本 result.csv (方便用 score_self.py 验证)
    result_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(result_dir, exist_ok=True)
    result_df = pd.DataFrame({'stock_id': v1_top5, 'weight': v1_weights})
    result_df.to_csv(os.path.join(result_dir, 'compare_v1.csv'), index=False)
    for label, stocks, weights, _ in v3_results:
        if stocks:
            result_df = pd.DataFrame({'stock_id': stocks, 'weight': weights})
            fname = f"compare_v3_{label.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_').replace('=', '_')}.csv"
            result_df.to_csv(os.path.join(result_dir, fname), index=False)
    print("\n已保存各版本到 output/")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
