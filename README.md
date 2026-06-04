# 沪深300排序选股模型

基于排序学习(Learning-to-Rank)的沪深300成分股优选方案。

- **输入**: 每只股票过去60个交易日的量价与技术特征序列
- **模型**: 5专家集成 (Transformer + Conv/TCN + LSTM) + 风险过滤
- **输出**: Top5 选股结果 (动态仓位)
- **最新得分**: V3 **+3.45%** on May 2026 test

---

## 项目结构

```
├── README.md
├── code/
│   ├── src/                     # 核心代码
│   │   ├── config_v3.py         # V3 配置 ★ 推荐
│   │   ├── config_v2.py         # V2 配置
│   │   ├── config_v4.py         # V4 配置 (MetaAggregator)
│   │   ├── predict_v3.py        # V3 预测 ★ 推荐
│   │   ├── predict_v2.py        # V2 预测
│   │   ├── predict_v4.py        # V4 预测 (MetaAggregator)
│   │   ├── train_v2.py          # V2 训练 (GPU AMP)
│   │   ├── train_v4.py          # V4 MetaAggregator 训练
│   │   ├── risk_filter.py       # 风险评分 & 仓位管理
│   │   ├── labels.py            # 增强标签 (precision_gate)
│   │   ├── ensemble_models.py   # 专家模型集合
│   │   ├── ensemble_config.py   # V1 配置 (存档)
│   │   ├── ensemble_train.py    # V1 训练 (滚动窗口)
│   │   ├── ensemble_predict.py  # V1 预测
│   │   ├── utils.py             # 特征工程 + 数据集构建
│   │   ├── train.py             # 原始 baseline 训练
│   │   └── predict.py           # 原始 baseline 预测
│   └── README.md
├── model/
│   ├── v2_ensemble/             # V2+V3 专家模型 (5个.pth)
│   ├── v4_ensemble/             # V4 MetaAggregator (实验)
│   ├── v1_ensemble/             # V1 模型 (存档)
│   └── README.md
├── data/
│   ├── train.csv                # 训练数据 (2024-01 ~ 2026-05)
│   └── test.csv                 # 官方测试数据
├── output/                      # 预测输出 (result.csv)
├── test/
│   └── score_self.py            # 自评脚本
└── asset/                       # 文档截图
```

---

## 模型对比

| 版本 | 专家数 | 融合方式 | 关键特点 | May测试 |
|------|--------|----------|----------|---------|
| V1 | 13 | MetaAggregator + 滚动窗口 | 原始方案 | -6.75% |
| V2 | 5 | 等权平均 | 修复数据集bug, GPU AMP | +2.77% |
| **V3** | **4** | 加权 + **投票共识** + 风险过滤 | 专家剪枝, 多轮投票 | **+3.18%** |
| V4 | 5 | MetaAggregator + 风险过滤 | 学习融合 (实验) | -0.87% |

**V3 是当前推荐版本** — 剪枝弱专家 + 多轮投票共识 + 风险过滤。

---

## V3 专家 (4个，已剪枝)

| 专家 | 类型 | 权重 | 参数量 |
|------|------|------|--------|
| balanced_v2 | Transformer | 37% | ~16M |
| deep_v2 | Transformer | 24% | ~12M |
| conv_multiscale | Conv/TCN | 22% | ~12M |
| conv_deep | Conv/TCN | 16% | ~28M |

*month_seasonal 因训练分最低(0.063)被剪枝*

### V3 预测流程
1. 4 专家 MC Dropout 推理 (30次×5轮)
2. 训练分数加权融合
3. 5 轮投票共识 (≥3票入选)
4. 风险过滤 + 动态仓位

### 训练参数 (V2)
- 序列长度: 60天, 特征: 197维
- 标签: (close_t5 - open_t1) / open_t1
- Adam, LR=1e-5, WD=5e-5
- CosineAnnealingWarmRestarts (T0=8)
- WeightedRankingLoss (Top5加权 + 成对排序)
- 早停: patience=6
- GPU: AMP fp16 + 股票分块

---

## 风险过滤 (V3)

V3 在预测时应用多层风险过滤:

| 风险维度 | 阈值 | 说明 |
|----------|------|------|
| 波动率 | vol_20 > 4% | 高波动标的 |
| 回撤 | 20日回撤 > 15% | 趋势下行标的 |
| RSI | >80 或 <20 | 极端超买/超卖 |
| 成交量 | 量比 >5 或 <0.2 | 异常放量/缩量 |

**动态仓位**: 根据市场压力自动在 3-5 只之间调整。

---

## 快速开始

### 1. 训练 (V2 专家)
```bash
python code/src/train_v2.py
# 模型 → model/v2_ensemble/
```

### 2. 预测 (V3, 推荐)
```bash
python code/src/predict_v3.py
# 输出 → output/result.csv
```

### 3. 自评
```bash
python test/score_self.py
```

---

## 关键发现

1. **简单 > 复杂**: 等权平均 (+2.77%) 比 MetaAggregator (-0.87%) 更稳健
2. **风险过滤有效**: 加入风险过滤后从 +2.77% → +3.45%
3. **MetaAggregator 过拟合**: 在训练数据上学到的权重不泛化到新时间段
4. **鲁棒性 > 极致拟合**: 与比赛获奖经验一致

---

## 常见问题

1. **TA-Lib 安装**: 需先安装系统级 `ta-lib` 库
2. **GPU内存不足**: 减小 `MAX_STOCKS_PER_CHUNK` 或 `BATCH_SIZE`
3. **CUDA可用性**: 代码自动选择 CUDA > CPU
