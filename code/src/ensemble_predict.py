"""
集成学习预测脚本
- 加载所有专家模型和元调度器
- MC Dropout 推理（亮度平均）
- 元调度器融合预测
- 输出 Top 5 股票到 output/result.csv
"""
import os
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

from ensemble_config import *
from ensemble_models import (
    StockTransformerExpert, ConvStockExpert, AdversarialStockExpert,
    MonthSeasonalExpert, MetaAggregator
)
from utils import engineer_features_39, engineer_features_158plus39
from predict import feature_cloums_map, feature_engineer_func_map


def create_expert_model(expert_cfg, input_dim, num_stocks):
    """根据配置创建专家模型"""
    exp_type = expert_cfg.get('type', 'transformer')

    if exp_type == 'transformer':
        return StockTransformerExpert(input_dim, expert_cfg, num_stocks)

    elif exp_type == 'conv':
        return ConvStockExpert(input_dim, expert_cfg, num_stocks)

    elif exp_type == 'adversarial':
        base_type = expert_cfg.get('base_type', 'transformer')
        if base_type == 'transformer':
            base_expert = StockTransformerExpert(input_dim, expert_cfg, num_stocks)
        else:
            base_expert = ConvStockExpert(input_dim, expert_cfg, num_stocks)

        d_model = expert_cfg.get('d_model', expert_cfg.get('hidden_channels', 256))
        adv_lambda = expert_cfg.get('adv_lambda', 0.1)
        num_domains = expert_cfg.get('num_time_domains', 12)
        return AdversarialStockExpert(base_expert, d_model, num_domains, adv_lambda)

    elif exp_type == 'month_seasonal':
        return MonthSeasonalExpert(input_dim, expert_cfg, num_stocks)

    else:
        raise ValueError(f"未知专家类型: {exp_type}")


def preprocess_predict_data(df, stockid2idx, feature_num):
    """预处理预测数据（与训练保持一致）"""
    assert feature_num in feature_engineer_func_map
    feature_engineer = feature_engineer_func_map[feature_num]
    feature_columns = feature_cloums_map[feature_num]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError('输入数据为空')

    num_procs = min(8, mp.cpu_count())
    with mp.Pool(processes=num_procs) as pool:
        processed_list = list(tqdm(
            pool.imap(feature_engineer, groups),
            total=len(groups), desc='特征工程'
        ))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
    """构建推理序列"""
    sequences, sequence_stock_ids = [], []
    for stock_id in stock_ids:
        stock_history = data[
            (data['股票代码'] == stock_id) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            seq = stock_history[features].values.astype(np.float32)
            sequences.append(seq)
            sequence_stock_ids.append(stock_id)

    if len(sequences) == 0:
        raise ValueError(f'没有可用于预测的股票序列')

    return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def main():
    # 设备检测
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"设备: {device}")

    # 1. 加载配置
    config_path = os.path.join(OUTPUT_DIR, 'ensemble_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到集成配置: {config_path}，请先运行训练")

    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    expert_configs = saved_config['expert_configs']
    num_stocks = saved_config['num_stocks']
    stockid2idx = saved_config['stockid2idx']
    feature_list = saved_config['feature_list']
    sequence_length = saved_config['sequence_length']
    feature_num = saved_config['feature_num']
    mc_samples = saved_config.get('mc_samples', MC_SAMPLES)
    input_dim = saved_config['input_dim']

    # 2. 加载数据
    data_file = os.path.join(DATA_PATH, 'train.csv')
    print(f"加载数据: {data_file}")
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()
    print(f"最新日期: {latest_date.date()}")

    # 3. 预处理
    stock_ids = sorted(raw_df['股票代码'].unique())
    idx2stock = {v: k for k, v in stockid2idx.items()}

    processed, features = preprocess_predict_data(raw_df, stockid2idx, feature_num)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 加载scaler
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        processed[features] = scaler.transform(processed[features])

    # 4. 构建推理序列
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed, features, sequence_length, stock_ids, latest_date
    )
    print(f"参与排序股票数: {len(sequence_stock_ids)}")

    x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]

    # 5. 加载所有专家模型
    print(f"\n加载 {len(expert_configs)} 个专家模型...")
    experts = []

    for exp_cfg in expert_configs:
        exp_name = exp_cfg['name']
        model_path = os.path.join(OUTPUT_DIR, f'expert_{exp_name}.pth')

        if not os.path.exists(model_path):
            print(f"  警告: 专家 {exp_name} 未找到模型文件，跳过")
            continue

        model = create_expert_model(exp_cfg, input_dim, num_stocks).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        experts.append(model)
        print(f"  已加载: {exp_name}")

    if len(experts) == 0:
        raise RuntimeError("没有加载到任何专家模型")

    # 6. 各专家 MC Dropout 预测
    print(f"\nMC Dropout 推理 (每个专家 {mc_samples} 次前向)...")
    all_expert_scores = []

    for i, expert in enumerate(experts):
        # MC Dropout 推理
        expert.train()  # 保持 dropout 激活
        mc_scores = []
        with torch.no_grad():
            for _ in range(mc_samples):
                if isinstance(expert, AdversarialStockExpert):
                    scores, _ = expert(x, return_features=True)
                else:
                    scores = expert(x)
                mc_scores.append(scores.squeeze(0))  # [N]

        avg_scores = torch.stack(mc_scores).mean(dim=0)  # 亮度平均
        all_expert_scores.append(avg_scores.cpu().numpy())
        print(f"  {expert_configs[i]['name']}: 完成")

    # 堆叠: [num_stocks, num_experts]
    expert_scores_np = np.stack(all_expert_scores, axis=-1)
    expert_scores_tensor = torch.from_numpy(expert_scores_np).unsqueeze(0).float().to(device)

    # 7. 元调度器融合
    meta_path = os.path.join(OUTPUT_DIR, 'meta_aggregator.pth')
    if os.path.exists(meta_path):
        print("加载元调度器...")
        meta = MetaAggregator(len(experts), num_stocks, hidden_dim=META_HIDDEN_DIM).to(device)
        meta.load_state_dict(torch.load(meta_path, map_location=device))
        meta.eval()

        with torch.no_grad():
            final_scores = meta(expert_scores_tensor).squeeze(0).cpu().numpy()
    else:
        # 无元调度器时简单平均
        print("未找到元调度器，使用简单平均融合")
        final_scores = expert_scores_np.mean(axis=-1)

    # 8. 排序输出 Top 5
    order = np.argsort(final_scores)[::-1]
    ranked_stock_ids = [sequence_stock_ids[i] for i in order]

    if len(ranked_stock_ids) < 5:
        raise ValueError(f'可预测股票不足5只，当前仅有 {len(ranked_stock_ids)} 只')

    top5 = ranked_stock_ids[:5]
    output_path = os.path.join('./output/', 'result.csv')
    output_df = pd.DataFrame({
        'stock_id': top5,
        'weight': [0.2] * len(top5),
    })
    os.makedirs('./output/', exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"预测日期: {latest_date.date()}")
    print(f"参与排序股票数: {len(ranked_stock_ids)}")
    print(f"Top 5 股票: {top5}")
    print(f"各专家分数分布:")
    for i, cfg in enumerate(expert_configs):
        top3_expert = np.argsort(all_expert_scores[i])[::-1][:3]
        print(f"  {cfg['name']}: Top3 = {[sequence_stock_ids[j] for j in top3_expert]}")
    print(f"结果已写入: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
