"""
优化版模型配置参数
- 序列长度、模型超参、训练参数、路径
- 相比原始baseline: batch_size↑, OneCycleLR, 早停, 多任务学习, 市场特征
"""
sequence_length = 60
feature_num = '158+39+market'
config = {
    'sequence_length': sequence_length,
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'batch_size': 4,            # 保持较小以避免CUDA OOM
    'num_epochs': 50,
    'learning_rate': 1e-4,      # OneCycleLR 的 max_lr
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    # 排序损失权重
    'pairwise_weight': 1.0,
    'base_weight': 1.0,
    'top5_weight': 2.0,

    # 多任务学习
    'multitask_weight': 0.3,    # 回归头损失权重

    # 标签类型: 'close_t5_open_t1' 修复前瞻偏差
    'label_type': 'close_t5_open_t1',

    # 早停
    'early_stopping_patience': 10,

    # 预测集成
    'ensemble_dates': 5,        # 取最近N个交易日做预测集成
    'ensemble_decay': 0.85,     # 指数衰减因子
    'turnover_penalty': 0.1,    # 换手惩罚（持有加成）

    # 市场特征开关
    'use_market_features': True,

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}
