"""
4维市场状态检测: 趋势/广度/加速跌/波动率
用于动态调整仓位数量和风控阈值
"""
import numpy as np
import pandas as pd


def compute_market_regime(data, feature_cols, stock_ids, reference_date):
    """
    计算4维市场状态信号 (0-1, 越高越危险)

    Returns:
        dict with:
        - trend_score: 趋势风险 (0-1)
        - breadth_score: 广度风险 (0-1)
        - accel_decline_score: 加速下跌风险 (0-1)
        - volatility_score: 波动率风险 (0-1)
        - composite: 综合市场压力 (0-1)
        - regime: 'risk_on' | 'neutral' | 'cautious' | 'risk_off'
    """
    result = {}

    # 计算市场等权收益
    daily_returns = {}
    for sid in stock_ids:
        stock_data = data[data['股票代码'] == sid].sort_values('日期')
        if 'return_1' in stock_data.columns:
            rets = stock_data[stock_data['日期'] <= reference_date]['return_1'].tail(30).dropna()
            if len(rets) >= 10:
                daily_returns[sid] = rets.values

    if len(daily_returns) < 10:
        return {'trend_score': 0.5, 'breadth_score': 0.5,
                'accel_decline_score': 0.5, 'volatility_score': 0.5,
                'composite': 0.5, 'regime': 'neutral'}

    ret_df = pd.DataFrame({k: pd.Series(v) for k, v in daily_returns.items()}).fillna(0)
    market_return = ret_df.mean(axis=1)

    # 1. 趋势风险
    ret_5d = market_return.tail(5).sum()
    ret_10d = market_return.tail(10).sum()
    ret_20d = market_return.tail(20).sum()

    trend_score = 0.0
    if ret_5d < -0.02: trend_score += 0.4
    elif ret_5d < -0.01: trend_score += 0.25
    if ret_10d < -0.03: trend_score += 0.35
    elif ret_10d < -0.015: trend_score += 0.2
    if ret_20d < -0.04: trend_score += 0.25
    result['trend_score'] = min(trend_score, 1.0)

    # 2. 广度风险 - 多少股票在MA以上
    above_ma5 = 0
    above_ma20 = 0
    total_valid = 0
    for sid in stock_ids:
        stock_data = data[(data['股票代码'] == sid) & (data['日期'] <= reference_date)].sort_values('日期')
        if '收盘' not in stock_data.columns:
            continue
        closes = stock_data['收盘'].tail(30)
        if len(closes) < 20:
            continue
        total_valid += 1
        if closes.iloc[-1] > closes.tail(5).mean():
            above_ma5 += 1
        if closes.iloc[-1] > closes.tail(20).mean():
            above_ma20 += 1

    if total_valid > 0:
        ratio_ma20 = above_ma20 / total_valid
        if ratio_ma20 < 0.2: breadth_score = 0.9
        elif ratio_ma20 < 0.35: breadth_score = 0.7
        elif ratio_ma20 < 0.5: breadth_score = 0.4
        elif ratio_ma20 < 0.65: breadth_score = 0.15
        else: breadth_score = 0.0
    else:
        breadth_score = 0.5
    result['breadth_score'] = breadth_score

    # 3. 加速下跌风险 - 近期跌速是否在加快
    accel_score = 0.0
    if len(market_return) >= 15:
        recent_5 = market_return.tail(5).sum()
        prior_10 = market_return.tail(15).head(10).sum()
        if recent_5 < 0 and prior_10 < 0 and recent_5 < prior_10 * 0.4:
            accel_score = 0.8
        elif recent_5 < 0 and recent_5 < prior_10 * 0.5:
            accel_score = 0.5
        elif recent_5 < -0.03:
            accel_score = 0.3
    result['accel_decline_score'] = accel_score

    # 4. 波动率风险
    vol_score = 0.0
    if len(market_return) >= 20:
        vol_20 = market_return.tail(20).std()
        vol_60 = market_return.tail(60).std() if len(market_return) >= 60 else vol_20
        vol_ratio = vol_20 / max(vol_60, 0.001)
        if vol_ratio > 2.0: vol_score = 0.9
        elif vol_ratio > 1.5: vol_score = 0.6
        elif vol_ratio > 1.2: vol_score = 0.3
        elif vol_ratio > 1.0: vol_score = 0.1
    result['volatility_score'] = vol_score

    # Diagnostic values (computed above, exposed for regime-conditional use)
    result['ret_5d'] = ret_5d
    result['ret_10d'] = ret_10d
    result['ret_20d'] = ret_20d
    result['ratio_ma20'] = ratio_ma20 if total_valid > 0 else 0.5
    result['vol_20'] = vol_20 if len(market_return) >= 20 else 0
    result['vol_60'] = vol_60 if len(market_return) >= 60 else result['vol_20']

    # 综合评分
    composite = (
        result['trend_score'] * 0.30 +
        result['breadth_score'] * 0.25 +
        result['accel_decline_score'] * 0.25 +
        result['volatility_score'] * 0.20
    )
    result['composite'] = composite

    # 市场状态分类
    if composite < 0.25:
        result['regime'] = 'risk_on'
    elif composite < 0.45:
        result['regime'] = 'neutral'
    elif composite < 0.72:
        result['regime'] = 'cautious'
    else:
        result['regime'] = 'risk_off'

    # 广度崩溃检测: <10% 股票在 MA20 以上 → 无条件空仓
    breadth_crash = breadth_score >= 0.95  # ratio < 0.1

    # 连续下跌熔断: 等权市场近5日每天都跌 → 等待企稳
    recent_5_returns = market_return.tail(5)
    consecutive_downs = (recent_5_returns < 0).sum() >= 5 if len(recent_5_returns) >= 5 else False

    # 加速下跌 + 趋势双高 → 无条件空仓 (即使波动率低也不能抄底)
    accel_trend_crash = (result['accel_decline_score'] >= 0.85 and result['trend_score'] >= 0.85)

    # 空仓决策
    skip_trading = (
        result['regime'] == 'risk_off' or
        breadth_crash or
        consecutive_downs or
        accel_trend_crash
    )
    result['accel_trend_crash'] = accel_trend_crash
    result['skip_trading'] = skip_trading
    result['breadth_crash'] = breadth_crash
    result['consecutive_downs'] = consecutive_downs

    # 根据状态调整参数
    if skip_trading:
        result['max_positions'] = 0
        result['min_positions'] = 0
    else:
        result['max_positions'] = _max_positions_for_regime(result['regime'])
        result['min_positions'] = _min_positions_for_regime(result['regime'])
    result['risk_threshold'] = _risk_threshold_for_regime(result['regime'])

    return result


def _max_positions_for_regime(regime):
    return {'risk_on': 5, 'neutral': 5, 'cautious': 3, 'risk_off': 0}[regime]


def _min_positions_for_regime(regime):
    return {'risk_on': 4, 'neutral': 3, 'cautious': 1, 'risk_off': 0}[regime]


def _risk_threshold_for_regime(regime):
    return {'risk_on': 90, 'neutral': 85, 'cautious': 75, 'risk_off': 60}[regime]
