"""
V4 配置: 5 专家 + MetaAggregator (实验)
- MetaAggregator: 学习专家融合权重
- 风险过滤在预测阶段应用
"""
import os

SEQUENCE_LENGTH = 60
FEATURE_NUM = '158+39'
INPUT_DIM = 197
NUM_STOCKS = 300
MC_SAMPLES = 20

EXPERT_CONFIGS = [
    {
        'name': 'balanced_v2',
        'type': 'transformer',
        'd_model': 256,
        'nhead': 4,
        'num_layers': 6,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.9,
    },
    {
        'name': 'deep_v2',
        'type': 'transformer',
        'd_model': 192,
        'nhead': 4,
        'num_layers': 8,
        'dim_feedforward': 384,
        'dropout': 0.1,
        'mc_dropout_rate': 0.12,
        'sd_prob': 0.85,
    },
    {
        'name': 'conv_multiscale',
        'type': 'conv',
        'hidden_channels': 256,
        'nhead': 4,
        'dropout': 0.12,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.9,
    },
    {
        'name': 'conv_deep',
        'type': 'conv',
        'hidden_channels': 384,
        'nhead': 4,
        'dropout': 0.15,
        'mc_dropout_rate': 0.12,
        'sd_prob': 0.85,
    },
    {
        'name': 'month_seasonal',
        'type': 'month_seasonal',
        'd_model': 128,
        'dropout': 0.1,
        'mc_dropout_rate': 0.08,
    },
]

BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-5
MAX_GRAD_NORM = 3.0
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 2e-4

PAIRWISE_WEIGHT = 1.0
TOP5_WEIGHT = 2.0
BASE_WEIGHT = 1.0

USE_AMP = True
MAX_STOCKS_PER_CHUNK = 25

META_HIDDEN_DIM = 64
META_EPOCHS = 20
META_LR = 1e-4

OUTPUT_DIR = './model/v4_ensemble'
V2_DIR = './model/v2_ensemble'
DATA_PATH = './data'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')

LABEL_TYPE = 'close_t5_open_t1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
