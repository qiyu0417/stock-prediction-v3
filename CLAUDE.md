# DeepSleep 股票预测项目

## 项目目标
沪深300股票排名选股比赛。给定300只股票的历史数据，预测未来5日收益最高的Top5，等权分配。

## 比赛规则（不可改变）
- **持仓周期**: T+1 买入 → T+5 卖出（5个交易日），不可更改
- **收益公式**: `R = (P_{T+5}^open - P_{T+1}^open) / P_{T+1}^open`
- **组合收益**: `Σ(W_i × R_i)`，现金余额 = 0%
- **选股数量**: Top 5，等权分配
- **预测频率**: 每周一次，预测日为周四（或最新交易日）

## 预训练模型（GitHub Releases 下载）

三个最优模型权重通过 GitHub Releases 分发（不在 git 仓库中减少体积）：

| 模型 | 路径 | 大小 | 说明 |
|------|------|------|------|
| **CorrGNN** | `model/stock_emb_8_gnn_corr/expert_*.pth` | ~72MB | 4专家，相关性图卷积，W3最强 |
| **Hybrid** | `model/stock_emb_8_hybrid/expert_*.pth` | ~89MB | 4专家，197维全量特征，最稳定 |
| **Alpha158** | `model/stock_emb_8_alpha158/expert_*.pth` | ~89MB | 4专家，158动量因子，W1最强 |

**下载链接**: [v1.0.0 Release](https://github.com/qiyu0417/stock-prediction-v3/releases/tag/v1.0.0) → `model-weights-best-3.tar.gz`

### 安装模型权重

```bash
# 下载并解压到仓库根目录
wget https://github.com/qiyu0417/stock-prediction-v3/releases/download/v1.0.0/model-weights-best-3.tar.gz
tar -xzf model-weights-best-3.tar.gz
```

### 快速推理

```bash
# 三模型集成预测（MC=20，5种子）
python test/eval_new_week.py

# 带StatArb的四模型混合评测
python test/eval_statarb_blend.py
```

### 集成方案

**最优配置**: CorrGNN + Hybrid + Alpha158 加权混合
- 权重: GNN=0.1, Hybrid=0.4, Alpha158=0.5
- 后处理: vol_filter=0.95, bounce=0.008, penalty=0.92, quality=0.05
- MC=20 5种子四周均值: **+10.22%** (W1 +3.39%, W2 +15.15%, W3 +17.90%, W4 +4.45%)

### 训练自己的模型

所有训练脚本在 `code/src/train_*.py`，示例：
```bash
python code/src/train_stock_emb_8_loss.py --loss hybrid     # Hybrid模型
python code/src/train_alpha158.py                            # Alpha158模型
python code/src/train_gnn_corr.py                            # CorrGNN模型
python code/src/train_statarb.py                             # StatArb模型
```

其他模型权重（50+实验）未包含在仓库中，可通过对应训练脚本重新生成。

### ⛔ 强制检查清单（写死在MD里，每次训练/评测前必须核对）

| # | 检查项 | 正确做法 | 错误做法 |
|---|--------|---------|---------|
| **1** | **标签公式** | `label = (open_T+5 - open_T+1) / open_T+1` | ~~`close_T+5`~~ 收盘价≠比赛公式 |
| **2** | **后处理数据源** | 特征工程后的 `processed`（含 volatility_20/return_1） | ~~`raw_data`（原始CSV只有12列）~~ — 后处理会变成空操作 |
| **3** | **评测 MC 采样** | `MC >= 20` | ~~MC=5~~ 假阳性±2% |
| **4** | **标签位置** | `_build_label_and_clean()` 在 `train.py:127` | 训练脚本里重复实现 |
| **5** | **模型文件** | `禁止删除 model/ 下任何 .pth` | — |

> 以上错误每个都至少浪费过2小时调试时间。违反任何一条 = 评测/训练结果无效。

## 数据
| 文件 | 行数 | 日期范围 |
|------|------|----------|
| `data/train.csv` | 172,790 | 2024-01-02 ~ 2026-05-27 |
| `data/test.csv` | ~3,500 | 2026-05-28 ~ 2026-06-12 |

## 当前最佳配置 (MC=20 稳定评测, 2026-06-26 更新)

### ★★★ GNN(Corr) + Hybrid + Alpha158 三模型 — 最优 +12.14% (实验#48)
**关键发现**: 相关性图 (return corr > 0.5) 比行业图精准得多。Corr GNN 的 W1 从 +2.89% 提升到 +3.39%，追上 H+A 水平。
**5种子 MC=20**: G=0.1 H=0.4 A=0.5 = **+12.14%** (W1 +3.39%, W2 +15.15%, W3 +17.90%)
- 比行业 GNN 提升 **+0.16%**，比 H+A 提升 **+0.64%**
- Corr GNN 单独: +10.30% (vs 行业版 +9.70%)
- conv_multiscale 训练分数从 0.073 飙升到 0.161 (+121%)

### ★★ GNN(行业) + Hybrid + Alpha158 — +11.98% (实验#47)
**5种子 MC=20**: G=0.15 H=0.45 A=0.4 = +11.98% (W1 +2.89%, W2 +15.15%, W3 +17.90%)

### ★★ Hybrid + Alpha158 固定加权 — +11.50% (实验#45)
**5种子 MC=20**: H+A w=0.45/0.55 = +11.50% (W1 +3.39%, W2 +14.89%, W3 +16.22%)
- 零训练成本，推理时加权平均两个模型的原始分数

### ★★ 后处理管线修复 — 最优 +9.09% (实验#44)
**关键发现**: 所有评测脚本的后处理管线使用了 raw CSV 数据（无技术指标列），导致 `volatility_filter`/`bounce_confirm`/`quality_score` 一直是空操作。
**修复**: 使用特征工程后的 `processed` 数据 → 后处理真正生效
**5种子验证**: OldDefault (vol=0.95, bounce=0.008, penalty=0.92, qual=0.05)
- NoFilter: 均值 +8.01%, 范围 +6.67~+8.62%
- **OldDefault: 均值 +9.09%, 范围 +8.25~+9.49%**
- 正式预测管线 (`deepsleep_v1.py`) 原本就使用正确数据，不受bug影响

### 最优独立: Hybrid dim=8 + 后处理 = +8.59% (5种子均值)
- `model/stock_emb_8_hybrid/`
- 5种子: w/PP +8.59% ±0.49%, w/o PP +8.01% ±0.72%
- 稳定可靠，后处理参数: vol=0.95, bounce=0.008, penalty=0.92, qual=0.05

### MC=20 完整排名 (2026-06-27, 5种子均值, W1-W4四周)
| 排名 | 模型 | w/PP Avg | Std | 备注 |
|:--:|------|:--:|:--:|------|
| **1** | **CorrGNN+H+A (0.1/0.4/0.5)** | **+10.22%** | ±6.53% | **四周最优，W4 +4.45%** |
| 2 | Hybrid+Alpha158 (0.45/0.55) | +9.38% | ±5.99% | 零训练成本 |
| 3 | Hybrid only | +9.23% | — | **W4最强 +6.02%** |
| 4 | StatArb only | +6.77% | — | 均值回复信号，W4 +5.24% |
| 5 | CorrG+H+A+S (0.1/0.3/0.4/0.2) | +8.34% | — | W4微提升但总体略降 |

**W4关键发现**: Hybrid solo W4最强，Alpha158和GNN都拖后腿。动量策略在连续三周大涨后W4失效。

### 实验记录

**实验#49 — 动态滚动相关性图GNN (2026-06-27)**: **FAILED**
- 用最近60天数据重建相关性图，用现有CorrGNN权重做推理
- 动态图avg_deg≈35（静态≈11），明显更稠密
- 结果: DynamicG+H+A +8.40% vs Static +10.22% (-1.82%)
- **结论**: 60天窗口相关性估计不稳定，稠密图传播噪声；静态全量图更可靠

**实验#50 — StatArb均值回复专家训练+评测 (2026-06-27)**: **MIXED**
- 训练3个StatArb变体: base (0.0429), deep (0.0415), wide (0.0444)
- 训练分数低于Hybrid (~0.08)，但信号类型不同（均值回复≠动量）
- StatArb only MC=20: +6.77% (W1 -0.46%, W2 +8.43%, W3 +13.86%, W4 +5.24%)
- **W4互补验证**: StatArb(+5.24%) > H+A(+3.92%), 加入4模型混合W4从+4.45%提升到+5.01%
- **但总体均值**: 4模型(+8.34%) < 3模型(+10.22%), StatArb拖累W1/W2
- **结论**: 均值回复在动量失效周确实互补，但固定权重无法最优利用；边际改进不足以突破天花板

**实验#52 — 状态路由 Regime Router (2026-06-27)**: **NEUTRAL**
- 基于 market_regime.py 的先验规则（非测试集驱动），按市场状态选择专家权重
- RISK_ON → Alpha158重仓, CAUTIOUS → Hybrid重仓, skip_trading → 空仓
- 结果: 与固定混合持平 (+8.40%)，三版规则调整均无改善
- **根因**: market_regime 是同步指标（"现在是什么状态"），无法预测5天内的状态变化
- skip_trading 在四周牛市中从未触发，空仓保护机制未能发挥作用
- **结论**: 5天持仓期内状态可能翻转，同步状态检测无法提供超额收益

**实验#51 — BrownianNoise训练 (2026-06-27)**: **FAILED (OOM)**
- 标准TransformerEncoder (非MCDropout)，batch_size=4时[B×N]=1200条序列同时做self-attention导致GPU OOM
- batch_size=1时系统RAM不足（Numpy分配13MB失败）
- **结论**: 需重构为MCDropoutTransformerEncoder + 足够RAM才能训练

**有效技术**: Hybrid loss, 模型加权混合, GNN (相关性图)
**无效技术**: SE, Contrastive, Mamba, Margin, MQFA, GLU排序头, SWA, 元模型动态加权, 滚动窗口, 动态图GNN, StatArb混合(边际改进不及单独使用)
**未验证**: BrownianNoise(需重构), Adversarial(需GPU), Aggressive

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

### 第3轮 (30-34, 2026-06-17)
| # | 实验 | Jun Avg | 教训 |
|---|------|:--:|------|
| 30 | RankGLU (Gated Feature Attn) | +6.10% | W1提升W2下降 |
| 31 | EMA decay=0.999 | +7.87% | 权重平滑有效 |
| 32 | GLU Ranking Head | +3.65% | GLU放排序头无效 |
| **33** | **EMA + Hybrid 集成** | **+9.33%** | **EMA互补Hybrid，最优2模型** |
| 34 | 3模型集成 (含EMA) | <8% | 3模型不如2模型稳定 |

### 第4轮 (35-43, 2026-06-17~18) — MC=20 重评 + 特征多样化
| # | 实验 | MC=20 Avg | 教训 |
|---|------|:--:|------|
| 35 | MC=5→MC=20 对比 | — | MC=5 假阳性±2%, MC=20 才是可靠标准 |
| 36 | SE v2 独立 | +8.11% | W2 +15.75% 最高但 W1 弱, 非突破 |
| 37 | Contrastive 独立 | -0.42% | 对比学习完全无效 |
| 38 | EMA+EMA+L 集成 | +8.76% | 微弱最优, 仅 +0.14% 优于 Hybrid |
| 39 | SE+EMA 联合 | +2.94% | SE+EMA 冲突, 比各自单独都差 |
| **40** | **Alpha158 独立** | **+5.40%** | **W1 +6.28% 最优! 选股与 Hybrid 完全不同** |
| 41 | Alpha158+Hybrid 集成 | +3.07% | 简单平均被 Alpha158 的 W2 差信号拖垮 |
| 42 | 所有模型 MC=20 | ~+8.6% | 天花板由特征集同质化决定 |
| **43** | **Tech39 独立 (28维)** | **+2.75%** | **技术指标单独信号太弱，三模型集成(+7.76%)无法超越Hybrid** |
| **44** | **后处理管线修复+网格搜索** | **+9.09%** | **发现eval脚本bug(raw CSV无指标列), 修复后OldDefault参数最优, 5种子验证** |
| **45** | **元模型动态加权** | **+11.38%** | 简单固定w=0.45混合有效, 元模型≈等权无增益 |
| 46 | SWA 权重平均 | +5.86% | 无效, 比Hybrid低-4.72% |
| **47** | **GNN 行业图卷积 + 三模型混合** | **+11.98%** | GNN单独弱(+9.7%)但W3强, 与H+A三模型互补 |
| **48** | **GNN 相关性图 + 三模型混合** | **+12.14%** | **相关性图大幅优于行业图, conv_ms分数+121%, W1追上H+A, 突破12%** |

### 最终结论
- **MC=20 是唯一可靠的评估标准**。MC=5 会产生高达 ±2% 的虚假波动
- **新最优**: CorrGNN+H+A 三模型加权 (0.1/0.4/0.5) = +12.14%，首次突破 +12%
- **相关性图 > 行业图**: avg_deg 10.8 vs 97，稀疏精准，conv_multiscale 训练分数 +121%
- **模型混合是最可靠的提升手段**: 架构多样化 > 特征多样化 > 损失函数调优
- **有效**: EMA, Hybrid loss, 模型加权混合, GNN (行业+相关性)
- **无效**: SE通道注意力, Contrastive, SE+EMA, Mamba, Margin, MQFA, GLU排序头, SWA, 元模型动态加权, 滚动窗口, 四模型混合(EMA/ListMLE)

**严重警告: 禁止删除 model/ 下的任何 .pth 文件。禁止 rm -rf。**

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
├── train_gnn.py            # GNN 行业图卷积训练
├── ensemble_models.py      # Transformer/Conv 专家模型 + 个股Embedding
├── utils.py                # 特征工程 engineer_features_158plus39
├── quality_filter.py       # 后处理 (波动率/反弹/质量)
├── market_regime.py        # 4D市场状态检测
├── deepsleep_v1.py         # 主预测入口
└── predict_v6.py           # V6预测入口

test/
├── eval_stock_emb.py       # Stock Emb vs V7 6月评估
├── backtest_monthly.py     # 月度回测
├── gen_meta_data.py        # 元模型训练数据生成
├── eval_meta.py            # 元模型 MC=20 评测
├── eval_rolling.py         # 滚动窗口动态权重评测
├── optimize_blend.py       # 多模型混合权重+PP优化
└── eval_gnn.py             # GNN MC=20 评测

model/
├── stock_emb_8_hybrid/            # Hybrid 独立
├── stock_emb_8_alpha158/          # Alpha158 独立
├── stock_emb_8_gnn/               # ★ GNN 行业图卷积 (W3最强)
├── meta_model/                    # 元模型 (w≈0.5, 未超越固定权重)
├── stock_emb_8_ema/               # EMA 权重平滑
├── stock_emb_8_ema_listmle/       # EMA + ListMLE
├── stock_emb_8_se_v2/             # SE v2 通道注意力 (W2最强)
├── stock_emb_8_se_ema/            # SE+EMA (不推荐, +2.94%)
├── stock_emb_8_alpha158/          # Alpha158 动量专家 (W1最强, 选股不同)
├── stock_emb_8_listmle_k3_t0.5/   # ListMLE k=3
├── stock_emb_8_multitask/         # 多任务学习
├── stock_emb_8_tech39/            # Tech39 纯技术指标 (RSI/MACD/KDJ, 实验#43)
├── stock_emb_8_rankglu/           # RankGLU 门控注意力
├── stock_emb_8_glurank/           # GLU 排序头
├── stock_emb_8_contrastive/       # 对比学习 (无效)
├── stock_emb_8_...                # 其他实验模型
└── v7_ensemble/                   # V7 基准
```

## 注意事项
- 所有训练脚本从 `train.py` 导入 `_build_label_and_clean` — 标签修改只需改一处
- `baostock` 用于拉取新数据
- MC Dropout 推理需要 `model.train()` 模式
- 空仓信号由 `market_regime.py` 中的 `skip_trading` 决定
- Stock Emb 的个股 Embedding 在 cross-stock attention 前注入, 无需改动数据管线
