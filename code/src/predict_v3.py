"""
V3 预测: 5 专家 MC Dropout + 分数加权融合 + 风险过滤 (推荐)
- 每个专家 30 次 MC Dropout 推理
- 训练分数加权融合
- 5 轮独立预测 → 投票选出最稳定的 Top5
- 风险过滤 + 动态仓位管理
"""
import os, sys, json, multiprocessing as mp
from collections import Counter
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_v3 import *
from ensemble_models import (
    StockTransformerExpert, ConvStockExpert, MonthSeasonalExpert
)
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from risk_filter import compute_risk_scores, apply_risk_filter


def _create_expert(cfg, input_dim, num_stocks):
    t = cfg.get('type', 'transformer')
    if t == 'transformer':
        return StockTransformerExpert(input_dim, cfg, num_stocks)
    elif t == 'conv':
        return ConvStockExpert(input_dim, cfg, num_stocks)
    elif t == 'month_seasonal':
        return MonthSeasonalExpert(input_dim, cfg, num_stocks)
    raise ValueError(f"未知类型: {t}")


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)

feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_predict(df, stockid2idx):
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]
    df = df.copy().sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    if not groups:
        raise ValueError('输入数据为空')
    processed = pd.concat([feature_engineer(g) for g in tqdm(groups, desc='预测集特征工程')]).reset_index(drop=True)
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
    if not sequences:
        raise ValueError(f'日期 {target_date} 无可用序列')
    return np.asarray(sequences, dtype=np.float32), seq_stock_ids


def run_one_round(experts, expert_weights, x, device, mc_samples, chunk_size, seed):
    """单轮 MC Dropout 推理 (确定性种子)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_amp = USE_AMP and device.type == 'cuda'
    all_scores = []
    for expert in experts:
        expert.train()
        mc_scores = []
        with torch.no_grad():
            for _ in range(mc_samples):
                with torch.amp.autocast('cuda', enabled=use_amp):
                    if x.size(1) <= chunk_size:
                        scores = expert(x).squeeze(0)
                    else:
                        chunk_scores = []
                        for start in range(0, x.size(1), chunk_size):
                            end = min(start + chunk_size, x.size(1))
                            sc = expert(x[:, start:end].contiguous()).squeeze(0)
                            chunk_scores.append(sc)
                        scores = torch.cat(chunk_scores, dim=0)
                mc_scores.append(scores)
        avg = torch.stack(mc_scores).mean(dim=0).cpu().numpy()
        all_scores.append(avg)
    fused = np.zeros(len(all_scores[0]))
    for w, scores in zip(expert_weights, all_scores):
        fused += w * scores
    return fused


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"设备: CPU")

    config_path = os.path.join(OUTPUT_DIR, 'ensemble_config.json')
    with open(config_path) as f:
        saved = json.load(f)

    expert_configs = saved['expert_configs']
    num_stocks = saved['num_stocks']
    stockid2idx = saved['stockid2idx']
    feature_list = saved['feature_list']

    # 专家权重 (前4个强专家，去掉 month_seasonal)
    # balanced:0.1855, deep:0.1215, conv_multi:0.1113, conv_deep:0.0804
    USE_EXPERT_INDICES = [0, 1, 2, 3]  # 只用前4个专家
    expert_configs = [expert_configs[i] for i in USE_EXPERT_INDICES]
    raw_scores = [0.1855, 0.1215, 0.1113, 0.0804]
    total = sum(raw_scores)
    expert_weights = [s / total for s in raw_scores]
    print(f"使用 {len(expert_configs)} 个专家 (去除 month_seasonal)")
    print(f"专家权重: {[f'{w:.3f}' for w in expert_weights]}")

    # 加载数据
    df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    latest_date = df['日期'].max()
    print(f"最新数据日期: {latest_date.date()}")

    stock_ids = sorted(df['股票代码'].unique())
    processed, features = preprocess_predict(df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    risk_scores_raw, market_stress = compute_risk_scores(
        processed, features, stock_ids, stock_ids,
        df[df['日期'] == df['日期'].max()]['日期'].iloc[0]
    )

    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        common = [c for c in scaler.feature_names_in_ if c in processed.columns]
        processed[common] = scaler.transform(processed[common])
        features = common

    sequences_np, seq_ids = build_sequences(processed, features, stock_ids, latest_date)
    print(f"参与排序股票: {len(seq_ids)}")

    risk_scores = {sid: risk_scores_raw.get(sid, 50) for sid in seq_ids}
    high_risk_count = sum(1 for s in risk_scores.values() if s > RISK_MAX_SCORE)
    print(f"市场压力: {market_stress:.2f}, 高风险股票: {high_risk_count}/{len(seq_ids)}")

    x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)

    # 加载专家 (只加载需要的)
    print(f"\n加载 {len(expert_configs)} 个专家...")
    experts = []
    for cfg in expert_configs:
        name = cfg['name']
        path = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
        model = _create_expert(cfg, len(features), num_stocks).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        experts.append(model)
        print(f"  {name}")

    # 多轮投票
    NUM_ROUNDS = 5
    MC_SAMPLES_PER_ROUND = 30
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999

    print(f"\nMC Dropout 投票 ({NUM_ROUNDS} 轮 × {MC_SAMPLES_PER_ROUND} 次前向)...")
    all_top5 = []
    all_fused = []

    for r in range(NUM_ROUNDS):
        fused = run_one_round(experts, expert_weights, x, device,
                            MC_SAMPLES_PER_ROUND, chunk_size, seed=42 + r * 100)
        all_fused.append(fused)

        # 本轮 Top5 (含风险过滤)
        selected, _ = apply_risk_filter(
            fused, seq_ids, risk_scores, market_stress,
            max_risk_score=RISK_MAX_SCORE,
            min_positions=RISK_MIN_POSITIONS,
            max_positions=RISK_MAX_POSITIONS,
        )
        all_top5.extend(selected)
        print(f"  第{r+1}轮 Top5: {selected}")

    # 投票统计
    vote_counts = Counter(all_top5)
    print(f"\n投票结果:")
    for stock, count in vote_counts.most_common(10):
        print(f"  {stock}: {count}/{NUM_ROUNDS} 票")

    # 平均分数 (用于最终排序)
    avg_fused = np.mean(all_fused, axis=0)
    std_fused = np.mean([np.std([all_fused[j][i] for j in range(NUM_ROUNDS)])
                         for i in range(len(seq_ids))])
    print(f"平均标准差: {std_fused:.5f}")

    # 最终选择: 得票 ≥ 3 的股票按平均分数排序
    consensus_stocks = [s for s, c in vote_counts.most_common() if c >= 3]
    if len(consensus_stocks) < 1:
        consensus_stocks = [s for s, _ in vote_counts.most_common(3)]

    # 按平均分数排序共识股票
    stock_to_idx = {s: i for i, s in enumerate(seq_ids)}
    consensus_scores = [avg_fused[stock_to_idx[s]] for s in consensus_stocks]
    sorted_consensus = [s for _, s in sorted(zip(consensus_scores, consensus_stocks), reverse=True)]

    # 最终仓位
    final_selected = sorted_consensus[:RISK_MAX_POSITIONS]
    final_weights = [1.0 / len(final_selected)] * len(final_selected)

    output_path = './output/result.csv'
    os.makedirs('./output/', exist_ok=True)
    pd.DataFrame({'stock_id': final_selected, 'weight': final_weights}).to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"市场压力: {market_stress:.2f} | 仓位: {len(final_selected)} 只")
    print(f"Top{len(final_selected)} (投票共识): {final_selected}")
    print(f"结果已写入: {output_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
