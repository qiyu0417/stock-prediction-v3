# DeepSleep — 沪深300排序选股模型

基于排序学习 (Learning-to-Rank) 的沪深300成分股优选方案。

**当前最优: ListMLE k=10 = 4专家集成 + 8维个股Embedding + ListMLE排序损失 + 后处理管线**

| 指标 | V7 (基准) | Stock Emb | ListMLE k=10 |
|------|------|------|------|
| 2026年6月样本外平均 | +5.15% | +5.54% | **+9.21%** |
| 6月W1 | +1.24% | +5.12% | +3.84% |
| 6月W2 | +9.06% | +5.96% | **+14.57%** |

---

## 模型架构

### 专家集成 (4 Experts)

| 专家 | 类型 | d_model | nhead | FFN | 层数 |
|------|------|---------|-------|-----|------|
| balanced_v3 | Transformer | 256 | 8 | 1024 | 6 |
| deep_v3 | Transformer | 192 | 8 | 768 | 8 |
| conv_multiscale | Conv/TCN | 256 | 4 | - | - |
| conv_deep | Conv/TCN | 384 | 4 | - | - |

- 等权平均融合 (0.25 × 4)
- MC Dropout 推理: 5轮 × 20次采样
- 输入: 60天 × 197维 (158 Alpha + 39 技术指标)
- Stock Emb 变体: 额外 8 维可学习个股 Embedding
- 损失函数: ListMLE (k=10, temperature=0.5) — 直接优化 Top-K 排序概率

### 后处理管线

```
MC Dropout 原始评分
    ↓
4D 市场状态检测 → 极端风险日空仓
    ↓
波动率过滤 → 剔除 top 5% 高波动股
    ↓
反弹确认 → 近2日未反弹降权 0.92×
    ↓
质量加分 → score += (quality - 0.5) × 0.05
    ↓
等权分配 Top5
```

### 4D 市场状态

| 维度 | 检测内容 | 权重 |
|------|----------|------|
| 趋势 | 5d/10d/20d 市场等权收益 | 30% |
| 广度 | 股价在 MA20 以上的股票占比 | 25% |
| 加速跌 | 近5日跌速 vs 前10日跌速 | 25% |
| 波动率 | vol_20 / vol_60 比率 | 20% |

---

## 项目结构

```
├── README.md
├── CLAUDE.md
├── code/src/
│   ├── deepsleep_v1.py         # 主预测入口
│   ├── predict_v6.py           # V6 预测管线
│   ├── config_v5.py              # V7 模型配置 (197维)
│   ├── config_stock_emb.py       # Stock Emb dim=4 配置
│   ├── config_stock_emb_8.py     # Stock Emb dim=8 配置
│   ├── ensemble_models.py        # 专家模型 (Transformer/Conv)
│   ├── quality_filter.py         # 后处理管线
│   ├── market_regime.py          # 4D 市场状态检测
│   ├── utils.py                  # 特征工程 (158+39+行业)
│   ├── train.py                  # 损失函数 + 数据集 (含ListMLE/ApproxNDCG/Hybrid)
│   ├── train_stock_emb_8_loss.py # dim=8 损失函数训练
│   ├── train_listmle_tune.py     # ListMLE k/temperature 调参
│   ├── train_v5_disk.py          # 磁盘管线训练
│   ├── train_stock_emb.py        # Stock Emb 训练
│   ├── fetch_industry.py       # 行业分类数据获取
│   ├── update_data.py          # 数据更新
│   ├── check_data.py           # 数据检查
│   └── stock_names.py          # 股票代码→名称映射
├── model/
│   ├── stock_emb_8_listmle_k10_t0.5/  # ★ 当前最优 (ListMLE k=10)
│   ├── stock_emb_ensemble/            # Stock Emb 基准 (WeightedRankingLoss)
│   ├── v7_ensemble/                   # V7 基准模型
│   ├── v5_ensemble/                   # V5 归档
│   └── v1_ensemble/                   # V1 归档
├── test/
│   ├── eval_stock_emb.py       # Stock Emb vs V7 评估
│   └── backtest_monthly.py     # 月度回测
├── data/
│   ├── train.csv               # 训练数据 (2024-01 ~ 2026-05)
│   ├── test.csv                # 测试数据 (2026-05-28 ~ 2026-06-12)
│   ├── industry.csv            # 行业分类
│   └── hs300_stock_list.csv    # 沪深300成分股
└── GUIDE.md                    # 环境搭建指南
```

---

## 快速开始

### 预测

```bash
python code/src/deepsleep_v1.py
python code/src/deepsleep_v1.py --date 2026-06-06
```

### 训练

```bash
# Stock Embedding 模型
python code/src/train_stock_emb.py

# V7 基准模型
python code/src/train_v5_disk.py
```

### 评估

```bash
# Stock Emb vs V7 6月对比
python test/eval_stock_emb.py

# 月度回测
python test/backtest_monthly.py
```

---

## 训练参数

| 参数 | 值 |
|------|-----|
| 序列长度 | 60 天 |
| 特征维度 | 197 (V7) / 197+4 个股Emb |
| 标签 | (close_t5 - open_t1) / open_t1 |
| Epochs | 50 |
| Batch Size | 4 |
| Learning Rate | 1e-5 |
| Weight Decay | 5e-5 |
| 优化器 | Adam |
| 调度器 | CosineAnnealingWarmRestarts (T0=8) |
| 损失函数 | WeightedRankingLoss / ListMLE (最优) |
| 早停 | patience=6, min_delta=2e-4 |

---

## 实验记录 (18次, 2次成功)

| # | 实验 | 结果 vs V7 | 结论 |
|---|------|:------:|------|
| 1 | 标签对齐 close→open | -4.35% | 收盘价信息 > 开盘价 |
| 2 | 剔除弱专家 4→3 | -8.31% | 集成多样性 > 个体质量 |
| 3 | 后处理网格搜索 | -5.23% | 历史最优 ≠ 未来最优 |
| 4 | CS截面特征 + LambdaRank | 全负分 | 截面特征编码方式有问题 |
| 5 | 8专家合并 (V7+V5) | V5死权重 | 专家质量参差不齐 |
| 6 | 6专家 (4基础+2变体) | 未完成 | 训练不稳定 |
| 7 | 训练窗口 (6m/12m/18m) | 全负 | 窗口不敏感 |
| 8 | 行业 one-hot (14维) | -3.55% | 14类太粗,制造业占一半 |
| 9 | 行业 Embedding (14→3) | -4.93% | 粗粒度行业信号是噪音 |
| **10** | **个股 Embedding (4维)** | **+0.39%** | **个股身份信息有效** |
| 11 | Embedding 8维 | - | 需配合更好损失函数 |
| **12** | **ListMLE k=10 T=0.5** | **+3.25%** | **排序损失 > 逐点回归损失** |
| 13 | LambdaRank 损失 | +0.48% | 微弱提升 |
| 14 | ApproxNDCG 损失 | -2.88% | 近似NDCG不稳定 |
| 15 | Hybrid 混合损失 | +1.85% | 混合不如纯ListMLE |
| 16 | k=5 vs k=10 对比 | k=10更优 | top-10容错空间有助于top-5 |
| 17 | 后处理参数网格搜索 | 无差异 | ListMLE对后处理不敏感 |
| 18 | 专家权重调优 | 无差异 | 等权已足够 |

### 关键教训

1. **不要改动后处理和集成策略** — V7 这层配置极其稳健
2. **排序损失 > 回归损失** — ListMLE 直接优化排序概率比逐点加权回归更匹配任务
3. **个股身份 > 行业分类** — 每只股票有自己的"个性"比粗行业标签有用
4. **等权融合已足够** — 专家权重调优无增益
