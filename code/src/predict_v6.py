"""
V6 预测管线: 4D市场状态 + 反弹确认 + 质量加分 + 置信度不等权 + 波动率过滤
使用 V5 已训练模型, 增强后处理
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from collections import Counter

from config_v5 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from market_regime import compute_market_regime
from quality_filter import (
    bounce_confirm, compute_quality_score,
    confidence_weighted_allocate, volatility_filter,
    BOUNCE_THRESHOLD
)


def load_v5_experts(feature_dim, num_stocks, device, model_dir='./model/v5_ensemble'):
    with open(os.path.join(model_dir, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    models = []
    for ec in cfg['expert_configs']:
        name = ec['name']
        path = os.path.join(model_dir, f'expert_{name}.pth')
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
    return models, [1.0 / len(models)] * len(models)


def mc_predict(experts, weights, x, device):
    """MC Dropout 推理 - 返回每只股票的原始评分"""
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'
    MC_SPR = 30
    NUM_ROUNDS = 5

    all_fused = []
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
        all_fused.append(fused)
    return np.mean(all_fused, axis=0)


def v6_predict(experts, weights, x, device, seq_ids,
               processed_data, feature_cols, reference_date,
               market_regime=None):
    """
    V6 增强预测管线:
    1. MC Dropout 获取原始评分
    2. 波动率过滤: 移除极端风险股
    3. 反弹确认: 仍在下跌的股票降低优先级
    4. 质量加分: 上调稳健趋势股
    5. 不等权分配: 高分配更多
    """
    # Step 1: 原始模型评分
    raw_scores = mc_predict(experts, weights, x, device)
    score_map = {sid: float(raw_scores[i]) for i, sid in enumerate(seq_ids) if i < len(raw_scores)}

    # Step 2: 波动率过滤 (只移除前5%极端值)
    filtered_ids = volatility_filter(processed_data, seq_ids, reference_date, top_pct=0.95)
    if len(filtered_ids) < 3:
        filtered_ids = seq_ids[:10]

    # Step 3: 反弹确认 (轻惩罚)
    confirmed = bounce_confirm(processed_data, filtered_ids, reference_date)
    for sid in filtered_ids:
        if sid not in confirmed:
            score_map[sid] = score_map.get(sid, 0) * 0.92

    # Step 4: 质量加分 (轻权重 5%)
    quality = compute_quality_score(processed_data, filtered_ids, reference_date)
    for sid in filtered_ids:
        if sid in score_map and sid in quality:
            q_bonus = quality[sid] - 0.5
            score_map[sid] += q_bonus * 0.05

    # 重新排序
    sorted_stocks = sorted(filtered_ids, key=lambda s: score_map.get(s, -999), reverse=True)

    # Step 5: 不等权分配
    if market_regime is None:
        market_regime = {'composite': 0.3, 'regime': 'neutral'}

    max_pos = market_regime.get('max_positions', 5)
    selected, weights = confidence_weighted_allocate(
        score_map, sorted_stocks, market_regime, max_positions=max_pos)

    return selected, weights
