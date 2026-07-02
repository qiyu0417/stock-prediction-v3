"""
Step 1: 预处理数据并保存到磁盘 (独立进程，做完就退出释放内存)
"""
import os, sys, gc, json
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np, pandas as pd, joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_stock_emb_8 import *
from train import (
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, set_seed, RankingDataset
)
from utils import create_ranking_dataset_vectorized
from train_stock_emb_8_loss import preprocess_with_winsor


def main():
    set_seed(42)
    print("Step 1: 加载原始数据...")
    full_df = pd.read_csv(TRAIN_FILE, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    print(f"  总数据: {len(full_df)} 行")

    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    print("Step 2: 特征工程 + Winsorization + StandardScaler...")
    processed, features, scaler, winsor_bounds = preprocess_with_winsor(full_df, stockid2idx)
    n_feats = len(features)
    print(f"  特征维度: {n_feats}")
    del full_df; gc.collect()

    print("Step 3: 构建排名数据集...")
    train_seq, train_tgt, train_rel, train_stk = create_ranking_dataset_vectorized(
        processed, features, SEQUENCE_LENGTH)
    print(f"  训练天数: {len(train_seq)}")
    del processed; gc.collect()

    # 保存到 disk
    cache_dir = os.path.join(DATA_PATH, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    print("Step 4: 保存预处理数据到磁盘...")
    np.save(os.path.join(cache_dir, 'train_seq.npy'), train_seq)
    np.save(os.path.join(cache_dir, 'train_tgt.npy'), train_tgt)
    np.save(os.path.join(cache_dir, 'train_rel.npy'), train_rel)
    np.save(os.path.join(cache_dir, 'train_stk.npy'), train_stk)
    joblib.dump(scaler, os.path.join(cache_dir, 'scaler.pkl'))
    with open(os.path.join(cache_dir, 'winsor_bounds.json'), 'w') as f:
        json.dump(winsor_bounds, f)
    with open(os.path.join(cache_dir, 'features.json'), 'w') as f:
        json.dump(features, f)
    with open(os.path.join(cache_dir, 'meta.json'), 'w') as f:
        json.dump({
            'n_feats': n_feats,
            'num_stocks': num_stocks,
            'n_days': len(train_seq),
            'stock_ids': stock_ids,
            'stockid2idx': stockid2idx,
        }, f)
    print(f"  已保存到 {cache_dir}/")
    print("Done! 现在可以运行训练脚本。")


if __name__ == '__main__':
    main()
