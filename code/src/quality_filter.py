"""
V6 选股增强: 反弹确认 + 质量加分 + 置信度不等权分配 + 波动率过滤
"""
import numpy as np
import pandas as pd


BOUNCE_THRESHOLD = 0.008  # 近2日收益 > 0.8% 取消谨慎


def bounce_confirm(data, stock_ids, reference_date):
    """
    反弹确认: 检查每只股票近2日是否反弹 > 0.8%
    返回: set of stock_ids that are NOT in danger (已反弹或无需谨慎)
    """
    confirmed = set()
    for sid in stock_ids:
        stock_data = data[(data['股票代码'] == sid) & (data['日期'] <= reference_date)].sort_values('日期')
        if 'return_1' not in stock_data.columns or len(stock_data) < 3:
            confirmed.add(sid)
            continue
        ret_2d = stock_data['return_1'].tail(2).sum()
        if ret_2d > BOUNCE_THRESHOLD:
            confirmed.add(sid)
    return confirmed


def compute_quality_score(data, stock_ids, reference_date):
    """
    质量加分: 偏向稳健趋势股
    维度:
      - 趋势一致性: 近20日正收益天数占比
      - 收益稳定性: 近20日日均收益 / 日收益标准差
      - 上升平滑度: 近20日收益 > 0 的天数中, 连续正收益天数

    返回: dict[sid] -> quality_score (0-1, 越高越好)
    """
    scores = {}
    for sid in stock_ids:
        stock_data = data[(data['股票代码'] == sid) & (data['日期'] <= reference_date)].sort_values('日期')
        if 'return_1' not in stock_data.columns or len(stock_data) < 22:
            scores[sid] = 0.5
            continue

        rets = stock_data['return_1'].tail(20).dropna()
        if len(rets) < 10:
            scores[sid] = 0.5
            continue

        ret_values = rets.values

        # 趋势一致性: 正收益天数占比
        up_days = (ret_values > 0).sum()
        consistency = up_days / len(ret_values)

        # 收益稳定性: mean / std (类似 Sharpe, 但用日数据)
        mean_ret = ret_values.mean()
        std_ret = ret_values.std()
        stability = mean_ret / max(std_ret, 0.001)

        # 上升平滑度: 最大连续正收益天数
        max_consecutive_up = 0
        current_streak = 0
        for r in ret_values:
            if r > 0:
                current_streak += 1
                max_consecutive_up = max(max_consecutive_up, current_streak)
            else:
                current_streak = 0

        smoothness = max_consecutive_up / max(len(ret_values), 1)

        # 综合评分
        quality = consistency * 0.4 + max(0, min(1, stability * 0.5)) * 0.35 + smoothness * 0.25
        scores[sid] = round(quality, 4)

    return scores


def confidence_weighted_allocate(scores, stock_ids, regime_info=None, max_positions=5,
                                  temperature=0.3, max_single=0.30, use_sigma=True):
    """
    不等权分配: Softmax温度 + 30%上限 + 可选σ仓位
    use_sigma=False 时只用 softmax+cap, 不根据置信度降低仓位
    """
    if not stock_ids:
        return [], []

    n = min(len(stock_ids), max_positions)
    selected = stock_ids[:n]

    raw_scores = np.array([scores.get(s, np.median(list(scores.values()))) for s in selected])
    all_scores = np.array(list(scores.values()))

    # Softmax权重
    shifted = raw_scores - raw_scores.max()
    weights = np.exp(shifted / temperature)
    weights = weights / weights.sum()

    # 单票上限
    for _ in range(10):
        overflow = 0.0
        capped = 0
        for i in range(n):
            if weights[i] > max_single:
                overflow += weights[i] - max_single
                weights[i] = max_single
                capped += 1
        if overflow > 0 and capped < n:
            uncapped = n - capped
            for i in range(n):
                if weights[i] < max_single:
                    weights[i] += overflow / uncapped
        else:
            break

    # 置信度σ仓位控制 (可选)
    if use_sigma:
        max_score = raw_scores.max()
        mean_score = all_scores.mean()
        std_score = all_scores.std()
        confidence = (max_score - mean_score) / (std_score + 1e-8)

        if confidence < 1.0:
            position_ratio = 0.30
        elif confidence < 2.0:
            position_ratio = 0.50 + (confidence - 1.0) * 0.25
        else:
            position_ratio = min(1.0, 0.75 + (confidence - 2.0) * 0.10)

        weights = weights * position_ratio

    return selected, weights.tolist()


def volatility_filter(data, stock_ids, reference_date, top_pct=0.85):
    """
    波动率过滤: 移除波动率在前 15% 的极端风险股

    返回: filtered stock_ids
    """
    volatilities = {}
    for sid in stock_ids:
        stock_data = data[(data['股票代码'] == sid) & (data['日期'] <= reference_date)].sort_values('日期')
        if 'volatility_20' in stock_data.columns:
            latest = stock_data['volatility_20'].dropna()
            if len(latest) > 0:
                volatilities[sid] = latest.iloc[-1]
            else:
                volatilities[sid] = 0.03
        else:
            volatilities[sid] = 0.03

    if len(volatilities) < 5:
        return stock_ids

    threshold = np.percentile(list(volatilities.values()), top_pct * 100)
    filtered = [sid for sid in stock_ids if volatilities[sid] <= threshold]
    return filtered
