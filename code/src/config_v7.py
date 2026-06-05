"""
V7 配置: V5 架构 + 100 epochs + 重新训练所有 4 专家 (含 Conv)
"""
import os

SEQUENCE_LENGTH = 60
FEATURE_NUM = '158+39'
INPUT_DIM = 197
NUM_STOCKS = 300
MC_SAMPLES = 20

EXPERT_CONFIGS = [
    {
        'name': 'balanced_v3',
        'type': 'transformer',
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'mc_dropout_rate': 0.1,
        'sd_prob': 0.85,
    },
    {
        'name': 'deep_v3',
        'type': 'transformer',
        'd_model': 192,
        'nhead': 8,
        'num_layers': 8,
        'dim_feedforward': 768,
        'dropout': 0.12,
        'mc_dropout_rate': 0.12,
        'sd_prob': 0.80,
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
]

BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-5
MAX_GRAD_NORM = 3.0
EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MIN_DELTA = 2e-4

PAIRWISE_WEIGHT = 1.0
TOP5_WEIGHT = 2.0
BASE_WEIGHT = 1.0

USE_AMP = True
MAX_STOCKS_PER_CHUNK = 50

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

RISK_MAX_SCORE = 90
RISK_MIN_POSITIONS = 3
RISK_MAX_POSITIONS = 5

OUTPUT_DIR = './model/v7_ensemble'
DATA_PATH = './data'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')

LABEL_TYPE = 'close_t5_open_t1'
