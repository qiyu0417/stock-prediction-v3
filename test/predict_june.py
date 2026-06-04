"""
V3 预测: 使用截至 2026-06-03 的全部数据，预测未来方向
输出 Top5 选股到 output/result.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
from collections import Counter

from config_v3 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from risk_filter import compute_risk_scores, apply_risk_filter

TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
TEST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'test.csv')
V3_EXPERT_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v2_ensemble')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


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


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"设备: {device}")

    # 合并 train.csv + test.csv 作为全量数据
    train_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码': str})
    test_df = pd.read_csv(TEST_PATH, dtype={'股票代码': str})
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    print(f"全量数据: {len(full_df)} 行, {full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}")

    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)
    print(f"股票数: {num_stocks}")

    # 特征工程
    print("\n特征工程...")
    processed, features = preprocess_for_date(full_df, stockid2idx)
    feature_dim = len(features)
    print(f"特征维度: {feature_dim}")

    # 标准化
    scaler_path = os.path.join(V3_EXPERT_DIR, 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    common = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[common] = scaler.transform(processed[common])

    # 风险评分
    risk_raw, stress = compute_risk_scores(
        processed, common, stock_ids, stock_ids,
        full_df['日期'].max()
    )
    print(f"市场压力: {stress:.1%}")

    # 构建序列 (截止 2026-06-03)
    pred_date = pd.to_datetime('2026-06-03')
    seq_np, seq_ids = build_sequences(processed, common, stock_ids, pred_date)
    print(f"可用股票: {len(seq_ids)}")

    risk_scores = {sid: risk_raw.get(sid, 50) for sid in seq_ids}
    x = torch.from_numpy(seq_np).unsqueeze(0).to(device)

    # 加载 V3 模型
    print("\n加载 V3 模型 (4 experts)...")
    v3_models = []
    v3_names = ['balanced_v2', 'deep_v2', 'conv_multiscale', 'conv_deep']
    v3_weights_raw = [0.1855, 0.1215, 0.1113, 0.0804]
    v3_weights = [w / sum(v3_weights_raw) for w in v3_weights_raw]

    for name in v3_names:
        path = os.path.join(V3_EXPERT_DIR, f'expert_{name}.pth')
        if name.startswith('conv'):
            cfg = {'name': name, 'type': 'conv',
                   'hidden_channels': 256 if 'multi' in name else 384,
                   'nhead': 4, 'dropout': 0.12 if 'multi' in name else 0.15,
                   'mc_dropout_rate': 0.1 if 'multi' in name else 0.12,
                   'sd_prob': 0.9 if 'multi' in name else 0.85}
            m = ConvStockExpert(feature_dim, cfg, num_stocks)
        else:
            cfg = {'name': name, 'type': 'transformer',
                   'd_model': 256 if name == 'balanced_v2' else 192, 'nhead': 4,
                   'num_layers': 6 if name == 'balanced_v2' else 8,
                   'dim_feedforward': 512 if name == 'balanced_v2' else 384,
                   'dropout': 0.1,
                   'mc_dropout_rate': 0.1 if name == 'balanced_v2' else 0.12,
                   'sd_prob': 0.9 if name == 'balanced_v2' else 0.85}
            m = StockTransformerExpert(feature_dim, cfg, num_stocks)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        v3_models.append(m)
        print(f"  {name}: {sum(p.numel() for p in m.parameters()):,} params")

    # V3 预测 (多轮投票 + 风险过滤)
    print("\n" + "=" * 50)
    print("V3 预测 (5轮投票 + 风险过滤)...")
    NUM_ROUNDS = 5
    MC_SPR = 30
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'

    all_top5 = []
    for r in range(NUM_ROUNDS):
        torch.manual_seed(42 + r * 100)
        np.random.seed(42 + r * 100)
        rnd_scores = []
        for expert in v3_models:
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
        for w, sc in zip(v3_weights, rnd_scores):
            fused += w * sc
        sel, _ = apply_risk_filter(fused, seq_ids, risk_scores, stress,
                                   max_risk_score=90, min_positions=3, max_positions=5)
        all_top5.extend(sel)

    vc = Counter(all_top5)
    consensus = [s for s, c in vc.most_common() if c >= 3]
    if len(consensus) < 1:
        consensus = [s for s, _ in vc.most_common(3)]
    if len(consensus) > 5:
        consensus = consensus[:5]

    print(f"\n  投票结果 (共 {NUM_ROUNDS * 5} 次选择):")
    for stock, count in vc.most_common(10):
        tag = " ***" if stock in consensus else ""
        print(f"    {stock}: {count}/{NUM_ROUNDS} 轮入选{tag}")

    # 输出
    weights = [1.0 / len(consensus)] * len(consensus)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = pd.DataFrame({'stock_id': consensus, 'weight': weights})
    result.to_csv(os.path.join(OUTPUT_DIR, 'result.csv'), index=False)

    print(f"\n{'='*50}")
    print(f"V3 Top{len(consensus)} (权重 {weights[0]:.2f})")
    print(f"{'='*50}")
    for i, (sid, w) in enumerate(zip(consensus, weights), 1):
        print(f"  {i}. {sid}  ({w:.2%})")

    # 查股票名称
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
    from stock_names import STOCK_NAMES
    print(f"\n{'='*50}")
    print("股票名称:")
    for sid in consensus:
        name = STOCK_NAMES.get(sid, '?')
        print(f"  {sid} → {name}")

    print(f"\n已保存 output/result.csv")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
