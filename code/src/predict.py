"""
优化版预测脚本
- 多日集成: 取最近N个交易日分别预测，指数加权融合
- 换手约束: 对已持有股票给予分数加成，降低不必要换手
- 输出Top5等权持仓到 output/result.csv
"""
import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39


_MARKET_COLS = ['market_return', 'market_up_ratio', 'market_volume_sum', 'market_volatility']

_BASE_COLS_39 = [
    'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
]

_ALPHA_158_COLS = [
    'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
    'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
    'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
    'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
    'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
    'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
    'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60',
    'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60',
    'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
    'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60',
    'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60',
    'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60',
    'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60',
    'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60',
    'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
    'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60',
    'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
    'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60',
    'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
    'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60',
    'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
    'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60',
    'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
    'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60',
    'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
    'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60',
    'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
    'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60',
    'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60',
    'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60',
    'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
]

_TECH_39_ONLY = [
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
    'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
    'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
    'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
]

feature_cloums_map = {
    '39': _BASE_COLS_39,
    '158+39': _ALPHA_158_COLS + _TECH_39_ONLY,
    '158+39+market': _ALPHA_158_COLS + _TECH_39_ONLY + _MARKET_COLS,
}

def _engineer_158plus39_market(df):
    return engineer_features_158plus39(df, add_market=True)


feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
    '158+39+market': _engineer_158plus39_market,
}


def preprocess_predict_data(df, stockid2idx):
    assert config['feature_num'] in feature_engineer_func_map, \
        f"Unsupported feature_num: {config['feature_num']}"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError('输入数据为空，无法预测')

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups),
                                   total=len(groups), desc='预测集特征工程'))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, target_date):
    """为指定日期构建所有股票的推理序列"""
    sequences, sequence_stock_ids, sequence_stock_indices = [], [], []
    for stock_id in stock_ids:
        stock_history = data[
            (data['股票代码'] == stock_id) & (data['日期'] <= target_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            sequences.append(stock_history[features].values.astype(np.float32))
            sequence_stock_ids.append(stock_id)
            instr_val = stock_history['instrument'].values[-1]
            sequence_stock_indices.append(int(instr_val))

    if len(sequences) == 0:
        raise ValueError(f'日期 {target_date} 没有可用于预测的股票序列')

    return np.asarray(sequences, dtype=np.float32), sequence_stock_ids, sequence_stock_indices


def predict_scores_for_date(model, sequences_np, stock_indices, device):
    """对单个日期执行预测，返回每只股票的得分"""
    x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)       # [1, N, L, F]
    idx = torch.LongTensor(stock_indices).unsqueeze(0).to(device)    # [1, N]
    with torch.no_grad():
        scores, _ = model(x, idx)  # [1, N], [1, N]
    return scores.squeeze(0).detach().cpu().numpy()


def select_with_turnover_penalty(scores, stock_ids, prev_holdings, penalty):
    """对已持有的股票给予分数加成，降低换手"""
    if not prev_holdings:
        order = np.argsort(scores)[::-1]
        return [stock_ids[i] for i in order[:5]]

    adjusted = scores.copy()
    id_to_idx = {sid: i for i, sid in enumerate(stock_ids)}
    for held in prev_holdings:
        if held in id_to_idx:
            adjusted[id_to_idx[held]] += penalty

    order = np.argsort(adjusted)[::-1]
    return [stock_ids[i] for i in order[:5]]


def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_path = os.path.join(config['output_dir'], 'best_model.pth')
    scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
    output_path = os.path.join('./output/', 'result.csv')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'未找到模型文件: {model_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---- 多日集成预测 ----
    all_dates = sorted(processed['日期'].unique())
    latest_date = all_dates[-1]
    seq_len = config['sequence_length']
    n_dates = config.get('ensemble_dates', 5)
    decay = config.get('ensemble_decay', 0.85)

    # 取最近 n_dates 个有足够历史数据的交易日
    min_idx = max(0, seq_len - 1)
    valid_dates = [d for d in all_dates if d >= all_dates[min_idx]]
    ensemble_dates = valid_dates[-n_dates:] if len(valid_dates) >= n_dates else valid_dates

    print(f"预测日期范围: {ensemble_dates[0].date()} ~ {ensemble_dates[-1].date()}")
    print(f"特征维度: {len(features)}")

    all_scores = []
    for i, date in enumerate(ensemble_dates):
        weight = decay ** (len(ensemble_dates) - 1 - i)
        try:
            seq_np, seq_ids, seq_indices = build_inference_sequences(
                processed, features, seq_len, stock_ids, date
            )
            day_scores = predict_scores_for_date(model, seq_np, seq_indices, device)
            all_scores.append((weight, day_scores, seq_ids))
        except ValueError as e:
            print(f"  跳过 {date.date()}: {e}")

    if not all_scores:
        raise ValueError('无法在任何日期生成预测')

    # 加权融合分数
    first_ids = all_scores[0][2]
    fused = np.zeros(len(first_ids))
    total_weight = 0
    for w, sc, ids in all_scores:
        if len(ids) == len(fused) and np.array_equal(ids, first_ids):
            fused += w * sc
            total_weight += w
        else:
            # 股票集合不同时，做对齐
            id_to_score = {sid: s for sid, s in zip(ids, sc)}
            aligned = np.array([id_to_score.get(sid, np.median(sc)) for sid in first_ids])
            fused += w * aligned
            total_weight += w
    fused /= total_weight

    # ---- 换手约束选股 ----
    prev_holdings = []
    prev_output = os.path.join(os.path.dirname(output_path), 'result_prev.txt')
    if os.path.exists(prev_output):
        with open(prev_output) as f:
            prev_holdings = [line.strip() for line in f if line.strip()]

    penalty = config.get('turnover_penalty', 0.1)
    top5 = select_with_turnover_penalty(fused, first_ids, prev_holdings, penalty)

    # 保存本次持仓供下次使用
    os.makedirs(os.path.dirname(prev_output), exist_ok=True)
    with open(prev_output, 'w') as f:
        for sid in top5:
            f.write(sid + '\n')

    # 输出
    output_df = pd.DataFrame({'stock_id': top5, 'weight': [0.2] * len(top5)})
    output_df.to_csv(output_path, index=False)

    print(f'最新数据日期: {latest_date.date()}')
    print(f'参与排序股票数: {len(first_ids)}')
    print(f'多日集成: {len(all_scores)} 个交易日')
    print(f'Top5: {top5}')
    print(f'结果已写入: {output_path}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
