"""
增强标签模块
参考比赛获奖经验: target_precision_gate
- 收益为正 + 排名前25% + 短期稳定 + 中期稳定 + 最大回撤<3%
"""
import numpy as np
import pandas as pd


def compute_precision_gate_labels(df, group_col='股票代码', date_col='日期'):
    """
    计算 target_precision_gate 标签。
    综合条件:
    1. 未来5日收益为正 (close_t5 > open_t1)
    2. 当日预测排名在全部股票中前25%
    3. 5日最大回撤 < 3%
    4. 短期和中期表现稳定

    Returns: DataFrame with added 'precision_gate' column (0/1)
    """
    df = df.copy()

    # 1. 计算未来5日收益排名 (需要在日期分组中做)
    df['rank_pct'] = df.groupby(date_col)['label'].transform(
        lambda x: x.rank(pct=True, ascending=False)
    )

    # 2. 计算5日最大回撤 (需要日内数据)
    df['dd_5d'] = df.groupby(group_col)['low'].transform(
        lambda x: x.rolling(5, min_periods=1).min()
    )
    df['max_5d'] = df.groupby(group_col)['high'].transform(
        lambda x: x.rolling(5, min_periods=1).max()
    )
    df['drawdown_5d'] = (df['dd_5d'] - df['max_5d'].shift(1)) / df['max_5d'].shift(1)

    # 3. 综合条件
    df['precision_gate'] = (
        (df['label'] > 0) &                          # 正收益
        (df['rank_pct'] >= 0.75) &                   # 排名前25%
        (df['drawdown_5d'].fillna(0) > -0.03)       # 回撤<3%
    ).astype(int)

    return df


def compute_weighted_label(df, alpha=0.7):
    """
    加权标签: alpha * precision_gate + (1-alpha) * normalized_label
    结合精确筛选和连续收益率
    """
    if 'precision_gate' not in df.columns:
        df = compute_precision_gate_labels(df)

    # 归一化原始label到 [0, 1]
    label_min = df['label'].min()
    label_max = df['label'].max()
    if label_max > label_min:
        df['label_norm'] = (df['label'] - label_min) / (label_max - label_min)
    else:
        df['label_norm'] = 0.5

    df['weighted_label'] = (alpha * df['precision_gate'] +
                            (1 - alpha) * df['label_norm'])
    return df


def compute_stability_score(df, group_col='股票代码'):
    """
    计算股票稳定性评分 (0-1)
    - 收益标准差低 → 稳定 → 高分
    - 最大回撤小 → 稳定 → 高分
    """
    # 5日收益波动率
    ret_std = df.groupby(group_col)['return_1'].transform(
        lambda x: x.rolling(10, min_periods=3).std()
    )

    # 稳定性评分
    if ret_std.std() > 0:
        stability = 1 - (ret_std - ret_std.min()) / (ret_std.max() - ret_std.min())
    else:
        stability = np.ones(len(df)) * 0.5

    df['stability_score'] = stability.fillna(0.5).clip(0, 1)
    return df


def create_multi_stage_labels(df, stage1_threshold=0.7, stage2_k=100):
    """
    多阶段标签:
    Stage 1: precision_gate = 1 → 候选池
    Stage 2: 候选池中按收益排序 → Top K
    Stage 3: 加入稳定性指标 → 最终排序
    """
    if 'precision_gate' not in df.columns:
        df = compute_precision_gate_labels(df)
    if 'stability_score' not in df.columns:
        df = compute_stability_score(df)

    # Stage 1 & 2: 精确门控 + 收益率
    df['stage1_pass'] = df['precision_gate'].astype(bool)
    df['stage2_rank'] = df.groupby('日期')['label'].transform(
        lambda x: x.rank(ascending=False)
    )
    df['stage2_pass'] = (df['stage1_pass'] &
                         (df['stage2_rank'] <= stage2_k))

    # Stage 3: 综合得分
    if 'label_norm' not in df.columns:
        label_min = df['label'].min()
        label_max = df['label'].max()
        df['label_norm'] = ((df['label'] - label_min) / (label_max - label_min)
                            if label_max > label_min else 0.5)

    df['final_score'] = (
        0.5 * df['label_norm'] +
        0.3 * df['precision_gate'] +
        0.2 * df['stability_score']
    )

    df['final_score'] = df['final_score'].fillna(0)

    return df
