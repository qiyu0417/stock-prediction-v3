"""
V2 Phase 2 配置: 5 专家 (2 Transformer + 2 Conv + 1 Seasonal)
- 修复: create_ranking_dataset_vectorized (299 stocks/day)
- 修复: LR=1e-5, WD=5e-5, GradNorm=3.0, Patience=6
- 新增: Conv + MonthSeasonal 专家增加多样性
"""
import os

SEQUENCE_LENGTH = 60
FEATURE_NUM = '158+39'
INPUT_DIM = 197
NUM_STOCKS = 300
MC_SAMPLES = 20

EXPERT_CONFIGS = [
    # ===== Transformer 专家 =====
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
    # ===== Conv 专家 (TCN) =====
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
    # ===== 月份季节性专家 =====
    {
        'name': 'month_seasonal',
        'type': 'month_seasonal',
        'd_model': 128,
        'dropout': 0.1,
        'mc_dropout_rate': 0.08,
    },
]

BATCH_SIZE = 4
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
MAX_STOCKS_PER_CHUNK = 50

OUTPUT_DIR = './model/v2_ensemble'
DATA_PATH = './data'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')

LABEL_TYPE = 'close_t5_open_t1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
