# model/ - 模型文件

| 目录 | 版本 | 专家数 | 融合方式 | May测试 | 状态 |
|------|------|--------|----------|---------|------|
| v2_ensemble/ | V2 | 5 | 等权平均 | +2.77% | 存档 |
| v2_ensemble/ + 风险过滤 | **V3** | 5 | 等权 + 风险过滤 | **+3.45%** | **推荐** |
| v4_ensemble/ | V4 | 5 | MetaAggregator + 风险过滤 | -0.87% | 实验 |
| v1_ensemble/ | V1 | 13 | MetaAggregator + 滚动窗口 | -6.75% | 存档 |
| 60_158+39/ | baseline | 1 | 单模型 | — | 存档 |

## 推荐使用

**V3** (V2 专家 + 风险过滤) 是当前最佳版本:
- 5 个 V2 预训练专家 (模型在 `v2_ensemble/`)
- 等权融合 (0.2 × 5)
- 风险过滤 + 动态仓位管理
- May 2026 测试得分: **+3.45%**

```bash
python code/src/predict_v3.py    # V3 预测
python test/score_self.py        # 自评
```

## 专家列表

| 专家 | 类型 | 参数量 | 大小 |
|------|------|--------|------|
| balanced_v2 | Transformer | ~16M | 19.6 MB |
| deep_v2 | Transformer | ~12M | 14.5 MB |
| conv_multiscale | Conv/TCN | ~12M | 14.5 MB |
| conv_deep | Conv/TCN | ~28M | 32.4 MB |
| month_seasonal | LSTM+Embed | ~0.5M | 0.6 MB |
