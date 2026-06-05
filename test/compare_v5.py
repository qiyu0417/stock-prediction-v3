"""
V3 vs V5 官方测试集对比 (test.csv: 2026-05-28 ~ 2026-06-03)
V5: nhead=8, FFN=4x Transformer + Winsorization
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
from collections import Counter

from config_v5 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from risk_filter import compute_risk_scores, apply_risk_filter

TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
TEST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'test.csv')
V3_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v2_ensemble')
V5_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v5_ensemble')

CUTOFF_DATE = '2026-05-27'


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_with_winsor(df, stockid2idx, winsor_bounds, scaler):
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    processed = pd.concat([feature_engineer(g) for g in tqdm(groups, desc='  特征工程', leave=False)]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    for col, (lo, hi) in winsor_bounds.items():
        if col in processed.columns:
            processed[col] = processed[col].clip(lo, hi)

    common = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[common] = scaler.transform(processed[common])
    return processed, common


def build_sequences(data, features, stock_ids, target_date):
    sequences, seq_stock_ids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= target_date)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            sequences.append(hist[features].values.astype(np.float32))
            seq_stock_ids.append(sid)
    return np.asarray(sequences, dtype=np.float32) if sequences else np.array([]), seq_stock_ids


def load_v3_experts(feature_dim, num_stocks, device):
    models = []
    names = ['balanced_v2', 'deep_v2', 'conv_multiscale', 'conv_deep']
    weights_raw = [0.1855, 0.1215, 0.1113, 0.0804]
    weights = [w / sum(weights_raw) for w in weights_raw]
    for name in names:
        path = os.path.join(V3_DIR, f'expert_{name}.pth')
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
        models.append(m)
    return models, weights


def load_v5_experts(feature_dim, num_stocks, device):
    with open(os.path.join(V5_DIR, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    expert_cfgs = cfg['expert_configs']

    models = []
    for ec in expert_cfgs:
        name = ec['name']
        path = os.path.join(V5_DIR, f'expert_{name}.pth')
        if not os.path.exists(path):
            continue
        if ec['type'] == 'transformer':
            m = StockTransformerExpert(feature_dim, ec, num_stocks)
        elif ec['type'] == 'conv':
            m = ConvStockExpert(feature_dim, ec, num_stocks)
        else:
            continue
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        models.append(m)
    # equal weights for now - can tune based on validation
    weights = [1.0 / len(models)] * len(models)
    return models, weights


def ensemble_predict(experts, weights, x, device, seq_ids, risk_scores, market_stress,
                     max_risk_score=85, min_positions=3, max_positions=5):
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


def calc_score(stock_ids, weights, test_data):
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

    # 加载数据
    full_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    test_df = pd.read_csv(TEST_PATH, dtype={'股票代码': str})
    test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
    test_df['日期'] = pd.to_datetime(test_df['日期'])
    test_stocks = set(test_df['股票代码'].unique())
    print(f"测试集: {len(test_df)} 行, {len(test_stocks)} 只股票, "
          f"{test_df['日期'].min().date()} ~ {test_df['日期'].max().date()}")

    train_df = full_df[full_df['日期'] <= CUTOFF_DATE].copy()
    stock_ids = sorted(test_stocks)
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # V3 预处理 (无 Winsorization)
    print("\nV3 预处理...")
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]
    df_v3 = train_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups_v3 = [g for _, g in df_v3.groupby('股票代码', sort=False)]
    processed_v3 = pd.concat([feature_engineer(g) for g in tqdm(groups_v3, desc='  特征工程')]).reset_index(drop=True)
    processed_v3['instrument'] = processed_v3['股票代码'].map(stockid2idx)
    processed_v3 = processed_v3.dropna(subset=['instrument']).copy()
    processed_v3['instrument'] = processed_v3['instrument'].astype(np.int64)
    processed_v3['日期'] = pd.to_datetime(processed_v3['日期'])

    scaler_v3 = joblib.load(os.path.join(V3_DIR, 'scaler.pkl'))
    common_v3 = [c for c in scaler_v3.feature_names_in_ if c in processed_v3.columns]
    processed_v3[common_v3] = scaler_v3.transform(processed_v3[common_v3])

    pred_date = pd.to_datetime(CUTOFF_DATE)
    seq_v3, seq_ids_v3 = build_sequences(processed_v3, common_v3, stock_ids, pred_date)
    risk_raw_v3, stress_v3 = compute_risk_scores(processed_v3, common_v3, stock_ids, stock_ids, train_df['日期'].max())
    risk_v3 = {sid: risk_raw_v3.get(sid, 50) for sid in seq_ids_v3}
    x_v3 = torch.from_numpy(seq_v3).unsqueeze(0).to(device)
    print(f"V3 可用: {len(seq_ids_v3)} 只")

    # V5 预处理 (Winsorization)
    print("\nV5 预处理 (Winsorization)...")
    with open(os.path.join(V5_DIR, 'winsor_bounds.json')) as f:
        winsor_bounds = json.load(f)
    scaler_v5 = joblib.load(os.path.join(V5_DIR, 'scaler.pkl'))

    processed_v5, common_v5 = preprocess_with_winsor(
        train_df, stockid2idx, winsor_bounds, scaler_v5)

    seq_v5, seq_ids_v5 = build_sequences(processed_v5, common_v5, stock_ids, pred_date)
    risk_raw_v5, stress_v5 = compute_risk_scores(processed_v5, common_v5, stock_ids, stock_ids, train_df['日期'].max())
    risk_v5 = {sid: risk_raw_v5.get(sid, 50) for sid in seq_ids_v5}
    x_v5 = torch.from_numpy(seq_v5).unsqueeze(0).to(device)
    print(f"V5 可用: {len(seq_ids_v5)} 只")

    del train_df, full_df, processed_v3, processed_v5, groups_v3; gc = __import__('gc'); gc.collect()

    # 加载模型
    print("\n加载模型...")
    v3_models, v3_w = load_v3_experts(len(common_v3), num_stocks, device)
    print(f"  V3: {len(v3_models)} experts")
    v5_models, v5_w = load_v5_experts(len(common_v5), num_stocks, device)
    print(f"  V5: {len(v5_models)} experts")

    # 预测
    risk_configs = [
        ('保守 (max=80)', 80, 1, 5),
        ('适中 (max=85)', 85, 3, 5),
        ('宽松 (max=90)', 90, 3, 5),
    ]

    print("\n" + "=" * 60)
    print("V3 预测...")
    v3_results = []
    for label, max_risk, min_pos, max_pos in risk_configs:
        top = ensemble_predict(v3_models, v3_w, x_v3, device, seq_ids_v3, risk_v3, stress_v3,
                               max_risk_score=max_risk, min_positions=min_pos, max_positions=max_pos)
        if len(top) > 5:
            top = top[:5]
        w = [1.0 / len(top)] * len(top) if top else []
        s = calc_score(top, w, test_df) if top else 0
        v3_results.append((label, top, s))
        print(f"  V3-{label}: {top} → {s:+.4%}")

    print("\n" + "=" * 60)
    print("V5 预测...")
    v5_results = []
    for label, max_risk, min_pos, max_pos in risk_configs:
        top = ensemble_predict(v5_models, v5_w, x_v5, device, seq_ids_v5, risk_v5, stress_v5,
                               max_risk_score=max_risk, min_positions=min_pos, max_positions=max_pos)
        if len(top) > 5:
            top = top[:5]
        w = [1.0 / len(top)] * len(top) if top else []
        s = calc_score(top, w, test_df) if top else 0
        v5_results.append((label, top, s))
        print(f"  V5-{label}: {top} → {s:+.4%}")

    # 汇总
    print("\n" + "=" * 60)
    print("V3 vs V5 对比 (测试集 2026-05-28 ~ 2026-06-03)")
    print("=" * 60)
    for (lbl, v3_stocks, v3_score), (_, v5_stocks, v5_score) in zip(v3_results, v5_results):
        diff = v5_score - v3_score
        flag = "▲" if diff > 0 else ("▼" if diff < 0 else "=")
        print(f"  {lbl}: V3 {v3_score:+.4%} vs V5 {v5_score:+.4%} ({flag} {diff:+.4%})")
        print(f"    V3: {v3_stocks}")
        print(f"    V5: {v5_stocks}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
