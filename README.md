# 沪深300选股模型 V3

基于排序学习 (Learning-to-Rank) 的沪深300成分股优选方案。

- **输入**: 每只股票过去 60 个交易日的量价与技术特征序列 (197 维)
- **模型**: 4 专家集成 (Transformer + Conv/TCN) + 多轮投票共识 + 风险过滤
- **输出**: Top 3-5 选股结果 (动态仓位)
- **最新得分**: **+4.13%** (2026-05-28 ~ 06-03 官方测试集)

---

## 项目结构

```
├── README.md
├── code/src/
│   ├── config_v3.py          # V3 配置 (推荐)
│   ├── predict_v3.py         # V3 预测 (推荐)
│   ├── risk_filter.py        # 风险评分 + 动态仓位
│   ├── config_v4.py          # V4 配置 (MetaAggregator 实验)
│   ├── predict_v4.py         # V4 预测
│   ├── train_v4.py           # V4 MetaAggregator 训练
│   ├── ensemble_models.py    # 专家模型定义 (Transformer/Conv/MetaAggregator)
│   ├── train.py              # 损失函数 + 标签构建
│   ├── utils.py              # 特征工程 + 数据集构建
│   └── labels.py             # 增强标签 (precision_gate)
├── model/
│   ├── v2_ensemble/          # V3 专家权重 (5个.pth + scaler)
│   └── v4_ensemble/          # V4 MetaAggregator (实验)
├── data/
│   ├── test.csv              # 官方测试数据 (2026-05-28 ~ 06-03)
│   └── hs300_stock_list.csv  # 沪深300 股票列表
├── test/
│   ├── predict_june.py       # 最新预测脚本
│   ├── compare_official.py   # V1 vs V3 官方测试集对比
│   ├── score_self.py         # 自评脚本
│   └── score_docker.py       # Docker 评分脚本
└── output/                   # 预测输出
```

---

## 版本演进

| 版本 | 专家数 | 融合方式 | 关键特点 | 官方测试 |
|------|--------|----------|----------|---------|
| V1 | 13 | MetaAggregator | 原始方案 | 不稳定 (-1.4% ~ +4.5%) |
| V2 | 5 | 等权平均 | 修复数据集 bug | +2.77% |
| **V3** | **4** | 加权 + **投票共识** + 风险过滤 | 专家剪枝, 多轮投票 | **+4.13%** |
| V4 | 5 | MetaAggregator + 风险过滤 | 学习融合 (实验) | -0.87% |

**V3 是推荐版本** — 剪枝弱专家 + 多轮投票共识 + 风险过滤。

---

## V3 专家 (4个，已剪枝)

| 专家 | 类型 | 权重 | 参数量 |
|------|------|------|--------|
| balanced_v2 | Transformer | 37% | ~16M |
| deep_v2 | Transformer | 24% | ~12M |
| conv_multiscale | Conv/TCN | 22% | ~12M |
| conv_deep | Conv/TCN | 16% | ~28M |

*month_seasonal 因训练分最低(0.063)被剪枝*

### 预测流程

1. 4 专家 MC Dropout 推理 (30次 × 5轮)
2. 训练分数加权融合
3. 5 轮投票共识 (≥3 票入选)
4. 风险过滤 + 动态仓位

### 训练参数

- 序列长度: 60 天, 特征: 197 维
- 标签: (close_t5 - open_t1) / open_t1
- Adam, LR=1e-5, WD=5e-5
- CosineAnnealingWarmRestarts (T0=8)
- WeightedRankingLoss (Top5 加权 + 成对排序)
- 早停: patience=6
- GPU: AMP fp16 + 股票分块

---

## 风险过滤

| 风险维度 | 阈值 | 说明 |
|----------|------|------|
| 波动率 | vol_20 > 4% | 高波动标的 |
| 回撤 | 20日回撤 > 15% | 趋势下行标的 |
| RSI | >80 或 <20 | 极端超买/超卖 |
| 成交量 | 量比 >5 或 <0.2 | 异常放量/缩量 |

**动态仓位**: 根据市场压力自动在 3-5 只之间调整。

---

## 快速开始

### 1. 预测 (V3)

```bash
git clone https://github.com/qiyu0417/stock-prediction-v3.git
cd stock-prediction-v3
pip install torch pandas numpy scikit-learn joblib tqdm
python code/src/predict_v3.py
# 输出 → output/result.csv
```

### 2. 自评

```bash
python test/score_self.py
```

### 3. 最新预测

```bash
python test/predict_june.py
```

---

## 关键发现

1. **简单 > 复杂**: 等权/加权平均比 MetaAggregator 更稳健
2. **风险过滤有效**: 加入风险过滤后从 +2.77% → +4.13%
3. **MetaAggregator 过拟合**: 在训练数据上学到的权重不泛化到新时间段
4. **投票共识消除随机性**: 5 轮投票 + 固定种子让 MC Dropout 结果可复现
5. **鲁棒性 > 极致拟合**: 与比赛获奖经验一致

---

## License

MIT
