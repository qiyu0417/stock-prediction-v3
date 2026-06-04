"""保存配置供预测使用"""
import sys, json, os
sys.path.insert(0, 'code/src')

import pandas as pd
from ensemble_config import *
from train import feature_cloums_map

# Rebuild stockid2idx from data
full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
stockids = sorted(full_df['股票代码'].unique())
stockid2idx = {s: i for i, s in enumerate(stockids)}
feature_list = feature_cloums_map[FEATURE_NUM]

cfg = {
    'sequence_length': SEQUENCE_LENGTH,
    'feature_num': FEATURE_NUM,
    'input_dim': len(feature_list),
    'expert_configs': EXPERT_CONFIGS,
    'num_stocks': len(stockids),
    'stockid2idx': stockid2idx,
    'feature_list': feature_list,
    'mc_samples': MC_SAMPLES,
}
with open(os.path.join(OUTPUT_DIR, 'ensemble_config.json'), 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print(f"配置已保存: {len(stockids)}股, {len(feature_list)}特征")
