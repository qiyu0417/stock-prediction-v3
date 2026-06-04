"""
V2 简化预测: 2 专家 MC Dropout + 加权平均
- 每个专家 20 次前向 (MC Dropout)
- 0.5/0.5 加权平均融合
- Top5 等权输出
"""
import os, sys, json, multiprocessing as mp
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_v2 import *
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
    """预测数据预处理"""
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
    """为指定日期构建所有股票的推理序列"""
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


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"设备: {gpu_name}")
    else:
        device = torch.device('cpu')
        print(f"设备: CPU")

    config_path = os.path.join(OUTPUT_DIR, 'ensemble_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置: {config_path}, 请先训练")

    with open(config_path) as f:
        saved = json.load(f)

    expert_configs = saved['expert_configs']
    num_stocks = saved['num_stocks']
    stockid2idx = saved['stockid2idx']
    feature_list = saved['feature_list']
    mc_samples = saved.get('mc_samples', MC_SAMPLES)
    expert_weights = saved.get('expert_weights', [0.5, 0.5])
    if len(expert_weights) != len(expert_configs):
        expert_weights = [1.0 / len(expert_configs)] * len(expert_configs)

    # 加载数据
    df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    latest_date = df['日期'].max()
    print(f"最新数据日期: {latest_date.date()}")

    stock_ids = sorted(df['股票代码'].unique())
    processed, features = preprocess_predict(df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 标准化前计算风险评分
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

    # 构建序列
    sequences_np, seq_ids = build_sequences(processed, features, stock_ids, latest_date)
    print(f"参与排序股票: {len(seq_ids)}")

    risk_scores = {sid: risk_scores_raw.get(sid, 50) for sid in seq_ids}
    high_risk_count = sum(1 for s in risk_scores.values() if s > 60)
    print(f"市场压力: {market_stress:.2f}, 高风险股票: {high_risk_count}/{len(seq_ids)}")

    x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)

    # 加载专家
    print(f"\n加载 {len(expert_configs)} 个专家...")
    experts = []
    for cfg in expert_configs:
        name = cfg['name']
        path = os.path.join(OUTPUT_DIR, f'expert_{name}.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到: {path}")
        model = _create_expert(cfg, len(features), num_stocks).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        experts.append(model)
        print(f"  已加载: {name}")

    # MC Dropout 推理 (AMP + 分块)
    print(f"\nMC Dropout 推理 ({mc_samples} 次前向)...")
    use_amp = USE_AMP and device.type == 'cuda'
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else len(seq_ids)
    all_scores = []
    for i, expert in enumerate(experts):
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
        print(f"  {expert_configs[i]['name']}: 完成")

    # 加权平均融合
    fused = np.zeros(len(seq_ids))
    for w, scores in zip(expert_weights, all_scores):
        fused += w * scores
    print(f"融合权重: {expert_weights}")

    # 风险过滤 + 动态仓位
    selected, weights = apply_risk_filter(
        fused, seq_ids, risk_scores, market_stress
    )

    output_path = './output/result.csv'
    os.makedirs('./output/', exist_ok=True)
    pd.DataFrame({'stock_id': selected, 'weight': weights}).to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"市场压力: {market_stress:.2f} | 仓位: {len(selected)} 只")
    print(f"Top{len(selected)}: {selected}")
    if len(selected) < 5:
        order = np.argsort(fused)[::-1]
        filtered_out = []
        for idx in order[:10]:
            sid = seq_ids[idx]
            if sid not in selected:
                filtered_out.append(f"{sid}(风险{risk_scores.get(sid, '?')})")
        if filtered_out:
            print(f"风险过滤: {', '.join(filtered_out[:5])}")
    print(f"结果已写入: {output_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
