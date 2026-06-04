"""
风险过滤 & 仓位管理模块
- 基于原始（未标准化）特征计算每只股票的风险评分
- 过滤高风险标的
- 根据市场状态动态调整仓位数量
"""
import numpy as np
import pandas as pd


def compute_risk_scores(df, features, stock_ids, seq_stock_ids, latest_date):
    """
    基于最近一个交易日的数据计算每只股票的风险评分。

    Args:
        df: 预处理后但未标准化的 DataFrame
        features: 可用特征列表
        stock_ids: 所有候选股票 ID
        seq_stock_ids: 有完整序列的股票 ID（与模型输出对应）
        latest_date: 最新日期

    Returns:
        risk_scores: dict[stock_id] -> risk_score (0-100, 越高越危险)
        market_stress: float (0-1, 市场整体压力水平)
    """
    recent = df[df['日期'] == latest_date].copy()
    if recent.empty:
        # 尝试日期字符串匹配
        recent = df[df['日期'].astype(str) == str(latest_date)[:10]].copy()
    if recent.empty:
        recent = df[df['日期'] == df['日期'].max()].copy()

    risk_scores = {}

    for sid in seq_stock_ids:
        stock_data = recent[recent['股票代码'] == sid]
        if stock_data.empty:
            risk_scores[sid] = 50  # 默认中等风险
            continue

        row = stock_data.iloc[0]
        score = 0

        # 1. 波动率风险 (0-25 分): volatility_20
        if 'volatility_20' in row.index:
            vol20 = float(row['volatility_20'])
            if vol20 > 0.04: score += 25
            elif vol20 > 0.03: score += 18
            elif vol20 > 0.02: score += 10

        # 2. 回撤风险 (0-25 分): 20日最低/最高比
        if 'MIN20' in row.index and 'MAX20' in row.index:
            max20 = max(float(row['MAX20']), 1e-8)
            min20_ratio = float(row['MIN20']) / max20
            if min20_ratio < 0.85: score += 25
            elif min20_ratio < 0.90: score += 18
            elif min20_ratio < 0.95: score += 10

        # 3. RSI 极端风险 (0-15 分)
        if 'rsi' in row.index:
            rsi = float(row['rsi'])
            if rsi > 80 or rsi < 20: score += 15
            elif rsi > 70 or rsi < 30: score += 8

        # 4. ATR 风险 (0-10 分)
        if 'atr_14' in row.index:
            atr = float(row['atr_14'])
            if atr > 0.06: score += 10
            elif atr > 0.04: score += 5

        # 5. 近期极端收益风险 (0-10 分)
        if 'return_5' in row.index:
            ret5 = abs(float(row['return_5']))
            if ret5 > 0.15: score += 10
            elif ret5 > 0.10: score += 5

        # 6. 成交量异常风险 (0-10 分)
        if 'volume_ratio' in row.index:
            vr = float(row['volume_ratio'])
            if vr > 5 or vr < 0.2: score += 10
            elif vr > 3 or vr < 0.3: score += 5

        # 7. 高低价差风险 (0-5 分)
        if 'high_low_spread' in row.index:
            hls = float(row['high_low_spread'])
            if hls > 0.08: score += 5

        risk_scores[sid] = min(score, 100)

    # 计算市场整体压力
    all_scores = list(risk_scores.values())
    if all_scores:
        stress = np.mean(all_scores) / 100.0
        high_risk_pct = sum(1 for s in all_scores if s > 50) / len(all_scores)
        market_stress = 0.5 * stress + 0.5 * high_risk_pct
    else:
        market_stress = 0.5

    return risk_scores, market_stress


def apply_risk_filter(model_scores, seq_stock_ids, risk_scores, market_stress,
                      max_risk_score=75, min_positions=3, max_positions=5):
    """
    应用风险过滤和动态仓位管理。

    Args:
        model_scores: 模型输出的预测分数 (np.array, shape [N])
        seq_stock_ids: 对应的股票 ID 列表
        risk_scores: 每只股票的风险评分 dict
        market_stress: 市场整体压力水平 (0-1)
        max_risk_score: 单只股票最大允许风险评分
        min_positions: 最少持仓数
        max_positions: 最多持仓数

    Returns:
        selected_stocks: 最终选出的股票列表
        weights: 对应的权重列表
    """
    # 1. 过滤高风险股票
    safe_mask = np.array([risk_scores.get(sid, 100) <= max_risk_score for sid in seq_stock_ids])

    filtered_scores = model_scores.copy()
    filtered_scores[~safe_mask] = -np.inf

    # 2. 根据市场压力调整最大持仓数
    if market_stress > 0.7:
        effective_max = max(min_positions, max_positions - 2)
    elif market_stress > 0.5:
        effective_max = max(min_positions, max_positions - 1)
    else:
        effective_max = max_positions

    # 3. 检查得分方差（模型能否有效区分股票）
    valid_scores = filtered_scores[filtered_scores > -np.inf]
    if len(valid_scores) >= 2:
        score_std = np.std(valid_scores)
        score_range = np.max(valid_scores) - np.min(valid_scores)
        # 得分区分度低 → 模型不确定 → 减少持仓
        if score_range < 0.001:
            effective_max = max(min_positions, effective_max - 2)
        elif score_range < 0.01:
            effective_max = max(min_positions, effective_max - 1)

    # 4. 选取 Top K
    order = np.argsort(filtered_scores)[::-1]

    selected = []
    for idx in order:
        sid = seq_stock_ids[idx]
        if filtered_scores[idx] > -np.inf:
            selected.append(sid)
        if len(selected) >= effective_max:
            break

    # 5. 生成权重 (等权)
    if not selected:
        return [], []

    weights = [1.0 / len(selected)] * len(selected)
    return selected, weights
