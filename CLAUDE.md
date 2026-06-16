# DeepSleep 股票预测项目

## 项目目标
沪深300股票排名选股比赛。给定300只股票的历史数据，预测未来5日收益最高的Top5，等权分配。

## 比赛收益公式
```
R = (P_{T+5}^open - P_{T+1}^open) / P_{T+1}^open
组合收益 = Σ(W_i × R_i)，现金余额 = 0%
等权: 每只股票投入相同金额
```

## 数据
| 文件 | 行数 | 日期范围 |
|------|------|----------|
| `data/train.csv` | 172,790 | 2024-01-02 ~ 2026-05-27 |
| `data/test.csv` | ~3,500 | 2026-05-28 ~ 2026-06-12 |

## 当前最佳配置: 2模型集成 ★ (2026-06-16 最终)

**集成方案**: Hybrid (dim=8, 50ep) + ListMLE k=3 T=0.5 (dim=8) — score平均后后处理
**6月表现**: Jun W1 +3.82%, Jun W2 +13.85%, 平均 **+8.83%**

### 模型1: Hybrid (dim=8, 50ep)
**位置**: `model/stock_emb_8_hybrid/`
**独立表现**: Jun W1 +2.93%, Jun W2 +13.55%, Avg +8.24%

### 模型2: ListMLE k=3 T=0.5 (dim=8)
**位置**: `model/stock_emb_8_listmle_k3_t0.5/`
**独立表现**: Jun W1 +2.50%, Jun W2 +8.83%, Avg +5.67%

### 单一最优模型: Hybrid dim=8 50ep
**配置**: 4专家 ensemble (config_stock_emb_8), 个股Embedding dim=8
- `balanced_v3`: Transformer d=256/8head/6层/FFN=1024
- `deep_v3`: Transformer d=192/8head/8层/FFN=768
- `conv_multiscale`: TCN hidden=256
- `conv_deep`: TCN hidden=384
- 融合: MC Dropout (5轮×20次)
- **个股Embedding**: 每只股票8维可学习向量, 在cross-stock attention前注入

**训练**: `code/src/train_stock_emb_8_loss.py --loss hybrid` — Hybrid loss, 50 epochs
**特征**: 197维 (158 Alpha + 39 技术指标) + 8维个股Embedding
**6月表现**: Jun W1 +2.93%, Jun W2 +13.55%, 平均 **+8.24%**

## 次优模型: ListMLE k=3 T=1.0

**位置**: `model/stock_emb_8_listmle_k3_t1.0/`
**6月表现**: Jun W1 +1.73%, Jun W2 +13.85%, 平均 +7.79%

## ListMLE k=10 (第三)

**位置**: `model/stock_emb_8_listmle_k10_t0.5/`
**6月表现**: Jun W1 +3.17%, Jun W2 +10.74%, 平均 +6.95%

## 基准: Stock Embedding (WeightedRankingLoss)

**位置**: `model/stock_emb_ensemble/`
**配置**: 同上但 dim=4, WeightedRankingLoss
**6月表现**: Jun W1 +5.12%, Jun W2 +5.96%, 平均 +5.54%

## 基准模型: V7 (DeepSleep V1)

**位置**: `model/v7_ensemble/`
**配置**: 同 Stock Embedding, 但不含个股Embedding, 纯197维技术面
**训练**: `code/src/train_v5_disk.py`
**特征**: 197维 (158 Alpha因子 + 39 技术指标), 60天序列

## 后处理管线 (`code/src/quality_filter.py`)
1. volatility_filter(top_pct=0.95) — 剔除波动率前5%
2. bounce_confirm(threshold=0.008) — 确认近2日反弹
3. 未确认反弹: score *= 0.92
4. quality_score: score += (quality - 0.5) * 0.05
5. 排序选Top5, equal_weight_allocate

## 市场状态 (`code/src/market_regime.py`)
4维评分(趋势/广度/加速下跌/波动率), risk_off>0.72 空仓

## 实际表现

### 6月样本外 (2026-06-01 ~ 2026-06-12)
| 周 | V7 | Stock Emb | k=10 T=0.5 | **k=3 T=0.5** |
|------|------:|------:|------:|------:|
| Jun W1 | +1.24% | +5.12% | +3.17% | +1.02% |
| Jun W2 | +9.06% | +5.96% | +10.74% | **+14.92%** |
| **平均** | +5.15% | +5.54% | +6.95% | **+7.97%** |

### 月度回测 (Jan-May 2026, 训练期内)
| 月 | V7 | Stock Emb |
|------|------:|------:|
| 1月 | -3.10% | +1.52% |
| 2月 | -2.49% | -7.15% |
| 3月 | +3.83% | -1.67% |
| 4月 | +3.39% | +1.56% |
| 5月 | +10.90% | +11.81% |
| **累计** | +12.48% | +5.25% |

## 全部实验记录 (29次)

### 前期 (1-19)
| # | 实验 | vs V7 | 教训 |
|---|------|:------:|------|
| 1-9 | 标签/专家/窗口/行业 | 全负 | — |
| **10** | **个股 Embedding (4维)** | **+0.39%** | **个股身份有效** |
| 12 | LambdaRank 损失 | +0.48% | 微弱提升 |
| **13** | **ListMLE k=10 T=0.5** | +1.80% | 排序损失有效 |
| 14 | ApproxNDCG 损失 | -2.88% | 不稳定 |
| 15 | Hybrid 损失 | +1.85% | 混合损失有效 |
| **19** | **ListMLE k=3 T=0.5** | **+2.43%** | k=3 > k=10 |

### 本次系统性搜索 (20-29, 2026-06-16)
| # | 实验 | Jun Avg | 教训 |
|---|------|:--:|------|
| 20 | dim=16+Hybrid | +5.26% | dim=16 过拟合 |
| 21 | 3模型集成 | +9.09% | 集成提升显著 |
| 22 | 100 epochs | +7.24% | 过拟合, 50ep更优 |
| 23 | k=2 T=0.5 | +6.28% | W1强W2弱 |
| 24 | CS 截面特征 Hybrid | +6.37% | 训练分数3x但过拟合 |
| 25 | k=1 T=0.5 | +4.53% | W1为负, k=3最优 |
| 26 | dim=4 Hybrid | +5.58% | 不如dim=8 |
| 27 | dim=32 Hybrid | -0.58% | 唯一负收益, dim=8最优 |
| **28** | **2模型集成** | **+8.83%** | **Hybrid+k3: 最精简最优** |

### 最终结论
- **最优**: 2模型集成 (Hybrid dim=8 + ListMLE k=3 T=0.5 dim=8)
- **Sweet spot**: dim=8, Hybrid loss, 50 epochs, k=3, T=0.5
- **无效**: dim缩放, CS特征, 更多epochs, 更多集成模型
- **集成规则**: 2个强模型 > 更多模型 (多样性不是越多越好)

**关键约束: 不要改动后处理和集成策略。**

**严重警告: 禁止删除 model/ 下的任何 .pth 文件。禁止 rm -rf。最优模型已设只读。**

## 关键文件
```
code/src/
├── config_v5.py            # V7 模型配置 (197维)
├── config_stock_emb_8.py   # Stock Emb dim=8 配置 (197+8维)
├── config_stock_emb.py     # Stock Emb 配置 (197+4维)
├── train.py                # _build_label_and_clean, ListMLELoss, ApproxNDCGLoss, HybridRankingLoss, RankingDataset
├── train_v5_disk.py        # V7 磁盘管线训练
├── train_stock_emb_8_loss.py  # Stock Emb dim=8 损失函数训练
├── train_listmle_tune.py   # ListMLE k/temperature 调参训练
├── train_stock_emb.py      # Stock Emb 训练
├── ensemble_models.py      # Transformer/Conv 专家模型 + 个股Embedding
├── utils.py                # 特征工程 engineer_features_158plus39
├── quality_filter.py       # 后处理 (波动率/反弹/质量)
├── market_regime.py        # 4D市场状态检测
├── deepsleep_v1.py         # 主预测入口
└── predict_v6.py           # V6预测入口

test/
├── eval_stock_emb.py       # Stock Emb vs V7 6月评估
└── backtest_monthly.py     # 月度回测

model/
├── stock_emb_8_listmle_k10_t0.5/  # ★ 当前最优 (ListMLE k=10)
├── stock_emb_8_listmle_k5_t1.0/   # ListMLE k=5
├── stock_emb_8_lambdarank/        # LambdaRank
├── stock_emb_8_approxndcg/        # ApproxNDCG
├── stock_emb_8_hybrid/            # Hybrid (部分训练)
├── stock_emb_ensemble/            # WeightedRankingLoss 基准
├── v7_ensemble/                   # V7 基准
├── v5_ensemble/                   # V5 归档
└── v1_ensemble/                   # V1 归档
```

## 注意事项
- 所有训练脚本从 `train.py` 导入 `_build_label_and_clean` — 标签修改只需改一处
- `baostock` 用于拉取新数据
- MC Dropout 推理需要 `model.train()` 模式
- 空仓信号由 `market_regime.py` 中的 `skip_trading` 决定
- Stock Emb 的个股 Embedding 在 cross-stock attention 前注入, 无需改动数据管线
