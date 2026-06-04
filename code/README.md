# code/src/ - 核心代码

## 基础模型 (Baseline)
| 文件 | 说明 |
|------|------|
| config.py | 原始baseline配置参数 |
| model.py | StockTransformer模型定义 |
| utils.py | 特征工程(39/158/197维) + 排序数据集构建 |
| train.py | 原始baseline训练脚本 |
| predict.py | 原始baseline预测脚本 |

## V1 集成模型 (存档)
| 文件 | 说明 |
|------|------|
| ensemble_config.py | V1集成学习总配置 (13专家+Meta) |
| ensemble_models.py | 专家模型集合 (Transformer/Conv/LSTM/MetaAggregator) |
| ensemble_train.py | V1集成训练 (滚动窗口+MetaAggregator) |
| ensemble_predict.py | V1集成预测 (MC Dropout+MetaAggregator) |

## V2 集成模型 (5专家等权)
| 文件 | 说明 |
|------|------|
| config_v2.py | V2配置 (5专家) |
| train_v2.py | V2训练 (全量数据, GPU AMP, 股票分块) |
| predict_v2.py | V2预测 (MC Dropout, 等权融合, 风险过滤) |

## V3 集成模型 ★ 推荐 (5专家等权 + 风险过滤)
| 文件 | 说明 |
|------|------|
| config_v3.py | V3配置 (使用V2专家, 风险过滤参数) |
| predict_v3.py | **V3预测** (等权融合 + 风险过滤) May: **+3.45%** |
| risk_filter.py | 风险评分 + 动态仓位管理 |
| labels.py | 增强标签 (precision_gate, 多阶段) |

## V4 集成模型 (5专家 + MetaAggregator, 实验)
| 文件 | 说明 |
|------|------|
| config_v4.py | V4配置 (MetaAggregator参数) |
| train_v4.py | V4训练 (基于V2专家训练MetaAggregator) |
| predict_v4.py | V4预测 (MetaAggregator + 风险过滤) May: -0.87% |

## 辅助脚本
| 文件 | 说明 |
|------|------|
| local_predict.py | 本地V1模型预测 |
| backtest_*.py | 历史回测脚本 |
| update_data.py | Baostock数据下载更新 |
| stock_names.py | 股票编码→名称映射 |
