"""
综合回测评估脚本
- Walk-forward 逐日回测：对测试期每一天独立预测Top5，跟踪5日实际收益
- 多维度对比：优化模型 vs 原始模型 vs 沪深300等权 vs 随机选股
- 输出指标：累计收益曲线、年化收益率、夏普比率、最大回撤、胜率、换手率
"""
import os
import sys
import warnings
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 将 src 目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import StockTransformer as StockTransformerOptimized
from utils import (
    engineer_features_158plus39,
    create_ranking_dataset_vectorized,
)


# ============================================================
# 配置
# ============================================================

DATA_PATH = './data'
ORIGINAL_MODEL_DIR = './model/60_158+39'
OPTIMIZED_MODEL_DIR = './model/60_158+39+market'
TEST_START = '2026-01-01'
TEST_END = '2026-05-31'
SEQUENCE_LENGTH = 60
TOP_K = 5
RISK_FREE_RATE = 0.02  # 无风险利率 2%

ORIGINAL_CONFIG = {
    'sequence_length': 60,
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'feature_num': '158+39',
}

OPTIMIZED_CONFIG = {
    'sequence_length': 60,
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'feature_num': '158+39+market',
}

_MARKET_COLS = ['market_return', 'market_up_ratio', 'market_volume_sum', 'market_volatility']

_TECH_39_ONLY = [
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
    'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
    'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
    'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
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


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# 原始 StockTransformer（兼容旧模型权重）
# ============================================================

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeatureAttention(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2), torch.nn.Tanh(),
            torch.nn.Linear(d_model // 2, 1), torch.nn.Softmax(dim=1)
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        w = self.attention(x)
        return self.dropout(torch.sum(x * w, dim=1))


class CrossStockAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attention = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, stock_features):
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        return self.norm(stock_features + self.dropout(attended))


class OriginalStockTransformer(torch.nn.Module):
    """与原始 baseline 完全一致的模型，用于加载旧权重"""
    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.num_stocks = num_stocks
        self.input_proj = torch.nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config['d_model'], nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'], dropout=config['dropout'], batch_first=True
        )
        self.temporal_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        self.cross_stock_attention = CrossStockAttention(config['d_model'], config['nhead'], config['dropout'])
        self.ranking_layers = torch.nn.Sequential(
            torch.nn.Linear(config['d_model'], config['d_model']),
            torch.nn.LayerNorm(config['d_model']), torch.nn.ReLU(), torch.nn.Dropout(config['dropout']),
            torch.nn.Linear(config['d_model'], config['d_model'] // 2),
            torch.nn.LayerNorm(config['d_model'] // 2), torch.nn.ReLU(), torch.nn.Dropout(config['dropout'])
        )
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            torch.nn.ReLU(), torch.nn.Dropout(config['dropout'] * 0.5),
            torch.nn.Linear(config['d_model'] // 4, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        src_r = src.view(batch_size * num_stocks, seq_len, feature_dim)
        src_p = self.pos_encoder(self.input_proj(src_r))
        tf = self.temporal_encoder(src_p)
        af = self.feature_attention(tf)
        sf = af.view(batch_size, num_stocks, -1)
        inf = self.cross_stock_attention(sf)
        inf = inf.view(batch_size * num_stocks, -1)
        rf = self.ranking_layers(inf)
        scores = self.score_head(rf)
        return scores.view(batch_size, num_stocks)


# ============================================================
# 特征工程辅助
# ============================================================

def _engineer_group_with_market(group_df):
    return engineer_features_158plus39(group_df, add_market=True)


def _engineer_group_no_market(group_df):
    return engineer_features_158plus39(group_df, add_market=False)


def preprocess_for_backtest(df, stockid2idx, feature_cols, add_market=True):
    """对全量数据做特征工程并标准化"""
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    num_proc = min(10, mp.cpu_count())
    engine_func = _engineer_group_with_market if add_market else _engineer_group_no_market
    with mp.Pool(processes=num_proc) as pool:
        processed_list = list(tqdm(
            pool.imap(engine_func, groups),
            total=len(groups), desc='特征工程'
        ))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument'])
    processed['instrument'] = processed['instrument'].astype(np.int64)

    # 只保留实际存在的列
    avail = [c for c in feature_cols if c in processed.columns]
    processed[avail] = processed[avail].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return processed, avail


# ============================================================
# 回测核心
# ============================================================

def run_backtest(model, processed, features, stock_ids, stockid2idx,
                 test_dates, sequence_length, device, model_type='optimized'):
    """
    Walk-forward 回测。
    对每个测试日，预测Top5，记录5日实际收益。
    """
    daily_results = []
    prev_holdings = []

    for date in tqdm(test_dates, desc=f'回测 ({model_type})'):
        # 构建当日推理序列
        seqs, seq_ids, seq_indices = [], [], []
        for sid in stock_ids:
            hist = processed[
                (processed['股票代码'] == sid) & (processed['日期'] <= date)
            ].sort_values('日期').tail(sequence_length)
            if len(hist) == sequence_length:
                seqs.append(hist[features].values.astype(np.float32))
                seq_ids.append(sid)
                seq_indices.append(int(hist['instrument'].values[-1]))

        if len(seqs) < TOP_K:
            continue

        seq_np = np.stack(seqs)
        x = torch.from_numpy(seq_np).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_type == 'optimized':
                idx = torch.LongTensor(seq_indices).unsqueeze(0).to(device)
                scores, _ = model(x, idx)
            else:
                scores = model(x)
        scores = scores.squeeze(0).cpu().numpy()

        # 换手约束
        adjusted = scores.copy()
        id_to_pos = {sid: i for i, sid in enumerate(seq_ids)}
        for held in prev_holdings:
            if held in id_to_pos:
                adjusted[id_to_pos[held]] += 0.1

        top_idx = np.argsort(adjusted)[::-1][:TOP_K]
        top_stocks = [seq_ids[i] for i in top_idx]

        # 计算5日后实际收益: (close_t5 - open_t1) / open_t1
        actual_returns = []
        for sid in top_stocks:
            fut = processed[
                (processed['股票代码'] == sid) & (processed['日期'] > date)
            ].sort_values('日期')
            if len(fut) >= 5:
                open_t1 = fut.iloc[0]['开盘']
                close_t5 = fut.iloc[4]['收盘']
                if open_t1 > 1e-4:
                    ret = (close_t5 - open_t1) / open_t1
                    actual_returns.append(ret)

        if len(actual_returns) < TOP_K:
            continue

        portfolio_return = np.mean(actual_returns)

        # 全市场平均收益（基准）
        all_rets = []
        for sid in stock_ids:
            fut = processed[
                (processed['股票代码'] == sid) & (processed['日期'] > date)
            ].sort_values('日期')
            if len(fut) >= 5:
                o1 = fut.iloc[0]['开盘']
                c5 = fut.iloc[4]['收盘']
                if o1 > 1e-4:
                    all_rets.append((c5 - o1) / o1)
        market_return = np.mean(all_rets) if all_rets else 0.0

        daily_results.append({
            'date': date,
            'portfolio_return': portfolio_return,
            'market_return': market_return,
            'top_stocks': top_stocks,
            'individual_returns': actual_returns,
        })

        prev_holdings = top_stocks

    return daily_results


def run_random_baseline(processed, stock_ids, test_dates, n_trials=10):
    """随机选股基准：多次试验取平均"""
    all_results = []
    for _ in range(n_trials):
        results = []
        for date in tqdm(test_dates, desc='随机基准', leave=False):
            available = []
            for sid in stock_ids:
                fut = processed[
                    (processed['股票代码'] == sid) & (processed['日期'] > date)
                ].sort_values('日期')
                if len(fut) >= 5:
                    o1 = fut.iloc[0]['开盘']
                    if o1 > 1e-4:
                        available.append(sid)

            if len(available) < TOP_K:
                continue

            chosen = np.random.choice(available, TOP_K, replace=False)
            rets = []
            for sid in chosen:
                fut = processed[
                    (processed['股票代码'] == sid) & (processed['日期'] > date)
                ].sort_values('日期')
                o1 = fut.iloc[0]['开盘']
                c5 = fut.iloc[4]['收盘']
                rets.append((c5 - o1) / o1)

            results.append(np.mean(rets))
        all_results.append(results)
    # 取平均
    max_len = max(len(r) for r in all_results)
    avg = []
    for i in range(max_len):
        vals = [r[i] for r in all_results if i < len(r)]
        avg.append(np.mean(vals) if vals else 0)
    return avg


# ============================================================
# 指标计算
# ============================================================

def compute_metrics(daily_returns, market_returns=None):
    """计算全套评估指标"""
    if not daily_returns:
        return {}

    rets = np.array(daily_returns)
    cum_ret = np.prod(1 + rets) - 1
    n_days = len(rets)
    ann_ret = (1 + cum_ret) ** (252 / max(n_days, 1)) - 1
    ann_vol = np.std(rets, ddof=1) * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / (ann_vol + 1e-12)
    win_rate = np.mean(rets > 0)

    # 最大回撤
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = np.min(drawdown)

    metrics = {
        'cumulative_return': cum_ret,
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'n_days': n_days,
        'mean_daily_return': np.mean(rets),
    }

    # 超额收益 vs 市场
    if market_returns and len(market_returns) == len(rets):
        mkt = np.array(market_returns)
        excess = rets - mkt
        metrics['excess_return'] = np.prod(1 + excess) - 1
        metrics['information_ratio'] = np.mean(excess) / (np.std(excess, ddof=1) + 1e-12) * np.sqrt(252)
        metrics['beat_market_ratio'] = np.mean(rets > mkt)

    return metrics


def compute_turnover(daily_results):
    """计算日均换手率"""
    if len(daily_results) < 2:
        return 0.0
    turnovers = []
    for i in range(1, len(daily_results)):
        prev = set(daily_results[i - 1]['top_stocks'])
        curr = set(daily_results[i]['top_stocks'])
        changed = len(curr - prev)
        turnovers.append(changed / TOP_K)
    return np.mean(turnovers) if turnovers else 0.0


# ============================================================
# 主程序
# ============================================================

def main():
    device = get_device()
    print(f"使用设备: {device}")

    # ---- 加载数据 ----
    data_file = os.path.join(DATA_PATH, 'train.csv')
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: i for i, sid in enumerate(stock_ids)}
    num_stocks = len(stock_ids)
    print(f"股票数量: {num_stocks}")

    # 确定测试日期
    all_dates = sorted(raw_df['日期'].unique())
    test_dates = [d for d in all_dates
                  if pd.Timestamp(TEST_START) <= d <= pd.Timestamp(TEST_END)]
    # 确保有足够历史 + 未来数据
    test_dates = [d for d in test_dates
                  if d >= all_dates[SEQUENCE_LENGTH - 1]
                  and d <= all_dates[-6]]  # 需要未来5天
    print(f"测试日期: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} 天)")

    # ---- 准备特征数据 ----
    print("\n===== 准备特征数据 =====")
    original_feature_cols = _ALPHA_158_COLS + _TECH_39_ONLY
    optimized_feature_cols = _ALPHA_158_COLS + _TECH_39_ONLY + _MARKET_COLS

    # 一份带市场特征，一份不带
    processed_opt, features_opt = preprocess_for_backtest(
        raw_df, stockid2idx, optimized_feature_cols, add_market=True
    )
    processed_orig, features_orig = preprocess_for_backtest(
        raw_df, stockid2idx, original_feature_cols, add_market=False
    )

    # 标准化 —— 必须使用与训练时相同的 scaler
    scaler_opt_path = os.path.join(OPTIMIZED_MODEL_DIR, 'scaler.pkl')
    scaler_orig_path = os.path.join(ORIGINAL_MODEL_DIR, 'scaler.pkl')

    if os.path.exists(scaler_opt_path):
        scaler_opt = joblib.load(scaler_opt_path)
        print("加载优化模型 scaler")
    else:
        scaler_opt = StandardScaler()
    processed_opt[features_opt] = scaler_opt.transform(processed_opt[features_opt])

    if os.path.exists(scaler_orig_path):
        scaler_orig = joblib.load(scaler_orig_path)
        print("加载原始模型 scaler")
    else:
        scaler_orig = StandardScaler()
    processed_orig[features_orig] = scaler_orig.transform(processed_orig[features_orig])

    # ---- 加载/训练模型 ----

    # 优化模型
    opt_model_path = os.path.join(OPTIMIZED_MODEL_DIR, 'best_model.pth')
    if os.path.exists(opt_model_path):
        print("\n===== 加载优化模型 =====")
        opt_model = StockTransformerOptimized(
            input_dim=len(features_opt), config=OPTIMIZED_CONFIG, num_stocks=num_stocks
        )
        opt_model.load_state_dict(torch.load(opt_model_path, map_location=device))
        opt_model.to(device)
        opt_model.eval()
    else:
        print(f"\n[警告] 优化模型未找到: {opt_model_path}")
        print("请先运行 train.py 训练优化模型，或修改 OPTIMIZED_MODEL_DIR")
        opt_model = None

    # 原始模型
    orig_model_path = os.path.join(ORIGINAL_MODEL_DIR, 'best_model.pth')
    if os.path.exists(orig_model_path):
        print("===== 加载原始模型 =====")
        orig_model = OriginalStockTransformer(
            input_dim=len(features_orig), config=ORIGINAL_CONFIG, num_stocks=num_stocks
        )
        orig_model.load_state_dict(torch.load(orig_model_path, map_location=device))
        orig_model.to(device)
        orig_model.eval()
    else:
        print(f"\n[警告] 原始模型未找到: {orig_model_path}")
        orig_model = None

    # ---- 执行回测 ----
    results = {}

    # 优化模型回测
    if opt_model is not None:
        print("\n===== 优化模型回测 =====")
        opt_results = run_backtest(
            opt_model, processed_opt, features_opt, stock_ids, stockid2idx,
            test_dates, SEQUENCE_LENGTH, device, model_type='optimized'
        )
        results['优化模型'] = opt_results
        results['优化模型_metrics'] = compute_metrics(
            [r['portfolio_return'] for r in opt_results],
            [r['market_return'] for r in opt_results]
        )
        results['优化模型_turnover'] = compute_turnover(opt_results)

    # 原始模型回测
    if orig_model is not None:
        print("\n===== 原始模型回测 =====")
        orig_results = run_backtest(
            orig_model, processed_orig, features_orig, stock_ids, stockid2idx,
            test_dates, SEQUENCE_LENGTH, device, model_type='original'
        )
        results['原始模型'] = orig_results
        results['原始模型_metrics'] = compute_metrics(
            [r['portfolio_return'] for r in orig_results],
            [r['market_return'] for r in orig_results]
        )
        results['原始模型_turnover'] = compute_turnover(orig_results)

    # 沪深300等权基准
    print("\n===== 沪深300等权基准 =====")
    market_returns = []
    for date in tqdm(test_dates, desc='等权基准'):
        all_rets = []
        for sid in stock_ids:
            fut = processed_orig[
                (processed_orig['股票代码'] == sid) & (processed_orig['日期'] > date)
            ].sort_values('日期')
            if len(fut) >= 5:
                o1 = fut.iloc[0]['开盘']
                c5 = fut.iloc[4]['收盘']
                if o1 > 1e-4:
                    all_rets.append((c5 - o1) / o1)
        market_returns.append(np.mean(all_rets) if all_rets else 0.0)
    results['沪深300等权'] = market_returns
    results['沪深300等权_metrics'] = compute_metrics(market_returns)

    # 随机选股基准
    print("\n===== 随机选股基准 =====")
    random_rets = run_random_baseline(processed_orig, stock_ids, test_dates, n_trials=10)
    results['随机选股'] = random_rets
    results['随机选股_metrics'] = compute_metrics(random_rets)

    # ---- 输出报告 ----
    print("\n" + "=" * 70)
    print("                    回 测 评 估 报 告")
    print("=" * 70)
    print(f"测试周期: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} 个交易日)")
    print(f"无风险利率: {RISK_FREE_RATE*100:.1f}%")

    print("\n" + "-" * 70)
    print(f"{'指标':<20} {'优化模型':>12} {'原始模型':>12} {'沪深300等权':>12} {'随机选股':>12}")
    print("-" * 70)

    metric_names = [
        ('cumulative_return', '累计收益率', '{:.2%}'),
        ('annualized_return', '年化收益率', '{:.2%}'),
        ('annualized_volatility', '年化波动率', '{:.2%}'),
        ('sharpe_ratio', '夏普比率', '{:.2f}'),
        ('max_drawdown', '最大回撤', '{:.2%}'),
        ('win_rate', '胜率', '{:.2%}'),
        ('mean_daily_return', '日均收益', '{:.4%}'),
        ('excess_return', '超额收益(vs市场)', '{:.2%}'),
        ('information_ratio', '信息比率', '{:.2f}'),
        ('beat_market_ratio', '跑赢市场比', '{:.2%}'),
    ]

    for key, name, fmt in metric_names:
        vals = []
        for label in ['优化模型', '原始模型', '沪深300等权', '随机选股']:
            m = results.get(f'{label}_metrics', {})
            v = m.get(key, None)
            if v is not None:
                vals.append(fmt.format(v))
            else:
                vals.append('N/A')
        print(f"{name:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

    print("-" * 70)
    # 换手率
    opt_to = results.get('优化模型_turnover')
    orig_to = results.get('原始模型_turnover')
    if opt_to is not None or orig_to is not None:
        opt_str = f'{opt_to:.1%}' if opt_to is not None else 'N/A'
        orig_str = f'{orig_to:.1%}' if orig_to is not None else 'N/A'
        print(f"{'日均换手率':<20} {opt_str:>12} {orig_str:>12} {'-':>12} {'-':>12}")

    print("=" * 70)

    # ---- 保存详细结果 ----
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # 汇总 CSV
    summary_rows = []
    for label in ['优化模型', '原始模型', '沪深300等权', '随机选股']:
        m = results.get(f'{label}_metrics', {})
        row = {'模型': label}
        row.update({k: m.get(k, None) for k, _, _ in metric_names})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, 'backtest_summary.csv'), index=False, encoding='utf-8'
    )

    # 每日收益 CSV
    daily_rows = []
    for label in ['优化模型', '原始模型']:
        daily_data = results.get(label, [])
        if isinstance(daily_data, list) and len(daily_data) > 0 and isinstance(daily_data[0], dict):
            for r in daily_data:
                daily_rows.append({
                    'date': r['date'],
                    'model': label,
                    'portfolio_return': r['portfolio_return'],
                    'market_return': r['market_return'],
                    'top_stocks': ','.join(r['top_stocks']),
                })
    if daily_rows:
        pd.DataFrame(daily_rows).to_csv(
            os.path.join(output_dir, 'backtest_daily.csv'), index=False, encoding='utf-8'
        )

    print(f"\n详细结果已保存到: {output_dir}/backtest_summary.csv")
    print(f"每日收益已保存到: {output_dir}/backtest_daily.csv")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
