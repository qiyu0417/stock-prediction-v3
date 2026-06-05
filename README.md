# DeepSleep V1 — 沪深300排序选股模型

基于排序学习 (Learning-to-Rank) 的沪深300成分股优选方案。

**DeepSleep V1 = 4专家集成 × 4D市场状态 × 反弹确认 × 质量加分 × 置信度不等权分配**

| 指标 | 数值 |
|------|------|
| 2026年1-5月累计收益 | **+13.00%** |
| 月胜率 | 3/5 (60%) |
| 最大单月收益 | +8.05% (1月) |
| 空仓月份 | 2/5 (2月、4月 — 正确避坑) |

---

## 模型架构

### 专家集成 (4 Experts)

| 专家 | 类型 | d_model | nhead | FFN | 层数 | 参数量 |
|------|------|---------|-------|-----|------|--------|
| balanced_v3 | Transformer | 256 | 8 | 1024 (4x) | 6 | ~16M |
| deep_v3 | Transformer | 192 | 8 | 768 (4x) | 8 | ~12M |
| conv_multiscale | Conv/TCN | 256 | 4 | - | - | ~12M |
| conv_deep | Conv/TCN | 384 | 4 | - | - | ~28M |

- 等权平均融合 (0.25 × 4)
- MC Dropout 推理: 5轮 × 30次采样
- 输入: 60天 × 197维 (158 Alpha + 39 技术指标)

### 后处理管线

```
MC Dropout 原始评分
    ↓
4D 市场状态检测 → 极端风险日空仓
    ↓
波动率过滤 → 剔除 top 5% 高波动股
    ↓
反弹确认 → 近2日未反弹 (>0.8%) 降权 0.92×
    ↓
质量加分 → 稳健趋势股获得 +5% 加成
    ↓
置信度不等权分配 → softmax(t=0.3) + 30% 单票上限
```

### 4D 市场状态

| 维度 | 检测内容 | 权重 |
|------|----------|------|
| 趋势 | 5d/10d/20d 市场等权收益 | 30% |
| 广度 | 股价在 MA20 以上的股票占比 | 25% |
| 加速跌 | 近5日跌速 vs 前10日跌速 | 25% |
| 波动率 | vol_20 / vol_60 比率 | 20% |

**空仓触发条件 (任一)**:
- 综合压力 > 0.65 (risk_off)
- < 25% 股票在 MA20 以上 (广度崩盘)
- 连续5天全部下跌
- 加速下跌 + 趋势双高

---

## 项目结构

```
├── README.md
├── code/src/
│   ├── deepsleep_v1.py        # DeepSleep V1 预测入口 ★
│   ├── config_v5.py            # 模型配置
│   ├── ensemble_models.py      # 专家模型定义 (Transformer/Conv/Meta)
│   ├── market_regime.py        # 4D 市场状态检测
│   ├── quality_filter.py       # 反弹确认 + 质量加分 + 不等权分配 + 波动率过滤
│   ├── train_v5_disk.py        # 磁盘管线训练脚本
│   ├── utils.py                # 特征工程 (158+39)
│   └── train.py                # 损失函数 + 数据集构建
├── model/v5_ensemble/          # DeepSleep V1 模型权重
├── test/
│   ├── compare_v6.py           # V6 (DeepSleep V1) 月度对比
│   ├── compare_v5.py           # V5 vs V3 对比
│   ├── compare_v7.py           # V6 vs V7 对比
│   └── compare_teammate.py     # 队友模型对比
├── data/
│   └── test.csv                # 官方测试数据
└── output/                     # 预测输出
```

---

## 快速开始

### 预测

```bash
# 预测 test.csv 日期
python code/src/deepsleep_v1.py

# 预测指定日期
python code/src/deepsleep_v1.py --date 2026-06-06

# 使用自定义数据
python code/src/deepsleep_v1.py --data path/to/data.csv --output path/to/result.csv
```

### 训练

```bash
# 两阶段磁盘管线 (避免 OOM)
python code/src/train_v5_disk.py
```

### 回测对比

```bash
# V6 / V7 / 队友模型 月度对比
python test/compare_v6.py
python test/compare_v7.py
python test/compare_teammate.py
```

---

## 版本演进

| 版本 | 专家数 | 关键改进 | 累计收益 |
|------|--------|----------|----------|
| V1 | 13 | MetaAggregator + 滚动窗口 | -6.75% |
| V2 | 5 | 数据集 bug 修复 | +2.77% |
| V3 | 4 | 专家剪枝 + 投票共识 | -0.81% |
| V5 | 4 | nhead=8, FFN 4x, Winsorization | +5.75% |
| **DeepSleep V1 (V6)** | **4** | **+4D市场状态 + 反弹确认 + 质量加分 + 不等权分配** | **+13.00%** |
| V7 | 4 | 100 epoch (过拟合) | +7.21% |

---

## 训练参数

| 参数 | 值 |
|------|-----|
| 序列长度 | 60 天 |
| 特征维度 | 197 (158 Alpha + 39 技术指标) |
| 标签 | (close_t5 - open_t1) / open_t1 |
| Epochs | 50 |
| Batch Size | 4 |
| Learning Rate | 1e-5 |
| Weight Decay | 5e-5 |
| 优化器 | Adam |
| 调度器 | CosineAnnealingWarmRestarts (T0=8) |
| 损失函数 | WeightedRankingLoss (listwise + pairwise + MSE) |
| 早停 | patience=6, min_delta=2e-4 |
| Winsorization | 1% / 99% 分位数截断 |

---

## 关键发现

1. **后处理 > 裸模型**: V5 裸模型 +5.75% → 加入 V6 管线 +13.00%
2. **4D 市场状态有效**: 1月 risk_on 赚 +8%, 2月/4月 correctly 空仓避坑
3. **简单融合 > 复杂融合**: 等权 0.25 优于 MetaAggregator
4. **50 epoch > 100 epoch**: 更多训练导致过拟合 (+7.21% vs +13.00%)
5. **轻后处理 > 重后处理**: 质量加分 5% / 反弹惩罚 0.92 优于更激进的参数
