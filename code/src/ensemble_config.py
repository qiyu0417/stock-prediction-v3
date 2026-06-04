"""
集成学习配置 - 滚动窗口 + MC Dropout + 专家模型 + 元调度器
"""
import os

# ============================================================
# 数据分割配置
# ============================================================
# 滚动窗口: 3个月一个窗口，随机取2个月训练+1个月验证
WINDOW_SIZE_MONTHS = 4      # 每窗口4个月
TRAIN_MONTHS = 3            # 每个窗口内训练月数
VAL_MONTHS = 1              # 每个窗口内验证月数
RANDOM_SPLIT_SEED = 42      # 月份随机分割种子

# ============================================================
# 序列与特征配置
# ============================================================
SEQUENCE_LENGTH = 60
FEATURE_NUM = '158+39'

# ============================================================
# 专家模型定义 - 不同架构风格
# ============================================================
EXPERT_CONFIGS = [
    # ========== Transformer 专家 ==========
    # 1. 深度专家: 多层小维度
    {
        'name': 'transformer_deep',
        'type': 'transformer',
        'd_model': 192,
        'nhead': 4,
        'num_layers': 6,
        'dim_feedforward': 384,
        'dropout': 0.15,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.9,
    },
    # 2. 宽度专家: 少层大维度
    {
        'name': 'transformer_wide',
        'type': 'transformer',
        'd_model': 512,
        'nhead': 8,
        'num_layers': 2,
        'dim_feedforward': 1024,
        'dropout': 0.15,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.9,
    },
    # 3. 平衡专家
    {
        'name': 'transformer_balanced',
        'type': 'transformer',
        'd_model': 256,
        'nhead': 4,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.12,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.9,
    },
    # 4. 深层注意力专家 (高dropout)
    {
        'name': 'transformer_attention',
        'type': 'transformer',
        'd_model': 320,
        'nhead': 8,
        'num_layers': 5,
        'dim_feedforward': 640,
        'dropout': 0.18,
        'mc_dropout_rate': 0.15,
        'sd_prob': 0.85,
    },
    # 5. 轻量快速专家
    {
        'name': 'transformer_lite',
        'type': 'transformer',
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'mc_dropout_rate': 0.08,
        'sd_prob': 0.92,
    },

    # ========== 卷积专家 (TCN风格) ==========
    # 6. 多尺度卷积专家
    {
        'name': 'conv_multiscale',
        'type': 'conv',
        'hidden_channels': 256,
        'nhead': 4,
        'dropout': 0.12,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.9,
    },
    # 7. 深卷积专家 (更大hidden，更多膨胀层)
    {
        'name': 'conv_deep',
        'type': 'conv',
        'hidden_channels': 384,
        'nhead': 4,
        'dropout': 0.15,
        'mc_dropout_rate': 0.12,
        'sd_prob': 0.85,
    },

    # ========== 对抗学习专家 (包装Transformer) ==========
    # 8. 对抗Transformer专家: 用GRL迫使特征时间不变
    {
        'name': 'adversarial_transformer',
        'type': 'adversarial',
        'base_type': 'transformer',
        'd_model': 256,
        'nhead': 4,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.12,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.9,
        'adv_lambda': 0.1,
        'num_time_domains': 12,
    },
    # 9. 对抗卷积专家: 卷积+对抗
    {
        'name': 'adversarial_conv',
        'type': 'adversarial',
        'base_type': 'conv',
        'hidden_channels': 256,
        'nhead': 4,
        'dropout': 0.12,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.9,
        'adv_lambda': 0.15,
        'num_time_domains': 12,
    },

    # ========== 月份季节性专家 ==========
    # 10. 股票×月份周期效应 (权重×1.5，强季节性因子)
    {
        'name': 'month_seasonal',
        'type': 'month_seasonal',
        'd_model': 128,
        'dropout': 0.1,
        'mc_dropout_rate': 0.08,
        'meta_weight': 1.5,  # 元调度器中的初始权重加成
    },

    # ========== 激进专家 (Aggressive) ==========
    # 11. 高风险高回报，大容量低dropout
    {
        'name': 'aggressive',
        'type': 'aggressive',
        'd_model': 512,
        'nhead': 8,
        'num_layers': 2,
        'dim_feedforward': 1024,
        'dropout': 0.05,
        'mc_dropout_rate': 0.05,
    },

    # ========== 布朗运动疯子专家 ==========
    # 12. 训练时注入布朗噪声，极端鲁棒
    {
        'name': 'brownian',
        'type': 'brownian',
        'd_model': 256,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.15,
        'mc_dropout_rate': 0.12,
        'noise_base': 0.02,
        'noise_max': 0.15,
    },

    # ========== GARCH统计套利回归专家 ==========
    # 13. 波动率建模+均值回复+截面残差
    {
        'name': 'statarb',
        'type': 'statarb',
        'd_model': 192,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 384,
        'dropout': 0.1,
        'mc_dropout_rate': 0.1,
        'statarb_lookback': 20,
    },
]

# 月份季节性专家权重（元调度器中给予更高初始权重）
MONTH_EXPERT_BOOST = 1.5

# 对抗学习配置
ADV_LAMBDA = 0.1           # 对抗损失权重（初始值）
ADV_LAMBDA_SCHEDULE = True  # 是否动态调整对抗权重

# ============================================================
# 训练配置
# ============================================================
BATCH_SIZE = 4
NUM_EPOCHS = 50              # 每个专家最大训练轮数
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-5          # 增大权重衰减防过拟合
MAX_GRAD_NORM = 3.0          # 更严格的梯度裁剪

# 早停配置
EARLY_STOPPING_PATIENCE = 6   # 连续N轮验证分数无提升则停止（更激进早停）
EARLY_STOPPING_MIN_DELTA = 2e-4  # 最小提升阈值

# 损失函数权重
PAIRWISE_WEIGHT = 1.0
TOP5_WEIGHT = 1.5           # 降低top5权重，避免过度聚焦少数样本
BASE_WEIGHT = 1.0

# MC Dropout 推理配置
MC_SAMPLES = 20              # 推理时前向传播次数（亮度平均）

# ============================================================
# 元调度器配置
# ============================================================
META_HIDDEN_DIM = 64         # 元调度器隐藏层维度
META_EPOCHS = 20             # 元调度器训练轮数
META_LR = 1e-4

# ============================================================
# 路径配置
# ============================================================
OUTPUT_DIR = './model/ensemble'
DATA_PATH = './data'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 设备与随机种子
# ============================================================
RANDOM_SEED = 42
DEVICE = 'cuda'  # 由代码自动检测覆盖
