"""
简化回测：直接对比优化模型 vs 原始模型
逐日预测Top5，追踪5日实际收益
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from config import config as opt_config
from model import StockTransformer as OptModel
from utils import engineer_features_158plus39
from sklearn.preprocessing import StandardScaler

SEQUENCE_LENGTH = 60
TOP_K = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 原始模型（兼容旧权重） ----
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])


class FeatureAttention(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2), torch.nn.Tanh(),
            torch.nn.Linear(d_model // 2, 1), torch.nn.Softmax(dim=1))
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x): return self.dropout(torch.sum(x * self.attention(x), dim=1))


class CrossStockAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attention = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, stock_features):
        a, _ = self.cross_attention(stock_features, stock_features, stock_features)
        return self.norm(stock_features + self.dropout(a))


class OrigModel(torch.nn.Module):
    """完全兼容原始 StockTransformer 权重"""
    def __init__(self, input_dim, cfg, num_stocks):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, cfg['d_model'])
        self.pos_encoder = PositionalEncoding(cfg['d_model'], cfg['dropout'], cfg['sequence_length'])
        el = torch.nn.TransformerEncoderLayer(cfg['d_model'], cfg['nhead'], cfg['dim_feedforward'], cfg['dropout'], batch_first=True)
        self.temporal_encoder = torch.nn.TransformerEncoder(el, cfg['num_layers'])
        self.feature_attention = FeatureAttention(cfg['d_model'], cfg['dropout'])
        self.cross_stock_attention = CrossStockAttention(cfg['d_model'], cfg['nhead'], cfg['dropout'])
        self.ranking_layers = torch.nn.Sequential(
            torch.nn.Linear(cfg['d_model'], cfg['d_model']), torch.nn.LayerNorm(cfg['d_model']), torch.nn.ReLU(), torch.nn.Dropout(cfg['dropout']),
            torch.nn.Linear(cfg['d_model'], cfg['d_model'] // 2), torch.nn.LayerNorm(cfg['d_model'] // 2), torch.nn.ReLU(), torch.nn.Dropout(cfg['dropout']))
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(cfg['d_model'] // 2, cfg['d_model'] // 4), torch.nn.ReLU(), torch.nn.Dropout(cfg['dropout'] * 0.5),
            torch.nn.Linear(cfg['d_model'] // 4, 1))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear): torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None: torch.nn.init.zeros_(m.bias)
    def forward(self, src):
        B, N, L, F = src.shape
        x = self.pos_encoder(self.input_proj(src.view(B * N, L, F)))
        x = self.temporal_encoder(x)
        x = self.feature_attention(x).view(B, N, -1)
        x = self.cross_stock_attention(x).view(B * N, -1)
        return self.score_head(self.ranking_layers(x)).view(B, N)

# ---- 特征工程 ----
def _feat_market(g):
    return engineer_features_158plus39(g, add_market=True)

def _feat_no_market(g):
    return engineer_features_158plus39(g, add_market=False)

def prepare_data(raw_df, stockid2idx, add_market):
    """对全量数据做特征工程"""
    df = raw_df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    import multiprocessing as mp
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    fn = _feat_market if add_market else _feat_no_market
    with mp.Pool(min(10, mp.cpu_count())) as pool:
        processed = pd.concat(list(tqdm(pool.imap(fn, groups), total=len(groups), desc='Feature eng'))).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument'])
    processed['instrument'] = processed['instrument'].astype(np.int64)
    return processed


def main():
    print(f"Device: {DEVICE}")

    # 加载数据
    raw_df = pd.read_csv('./data/train.csv', dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # 测试日期
    all_dates = sorted(raw_df['日期'].unique())
    test_start = pd.Timestamp('2026-01-05')
    test_end = pd.Timestamp('2026-02-27')
    test_dates = [d for d in all_dates if test_start <= d <= test_end and d >= all_dates[59] and d <= all_dates[-6]]
    print(f"Test period: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} days)")

    # ---- 准备优化模型数据 ----
    print("\nPreparing optimized model data...")
    processed_opt = prepare_data(raw_df, stockid2idx, add_market=True)
    opt_feature_names = [c for c in processed_opt.columns if c not in ['股票代码', '日期']]
    opt_scaler = joblib.load('./model/60_158+39+market/scaler.pkl')
    # 确保特征顺序与训练时一致
    expected_features = list(opt_scaler.feature_names_in_)
    common = [c for c in expected_features if c in processed_opt.columns]
    opt_feature_names = common
    print(f"Optimized features (aligned): {len(opt_feature_names)}")
    processed_opt[opt_feature_names] = processed_opt[opt_feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed_opt[opt_feature_names] = opt_scaler.transform(processed_opt[opt_feature_names])

    # ---- 准备原始模型数据 ----
    print("Preparing original model data...")
    processed_orig = prepare_data(raw_df, stockid2idx, add_market=False)
    orig_scaler = joblib.load('./model/60_158+39/scaler.pkl')
    expected_features_orig = list(orig_scaler.feature_names_in_)
    common_orig = [c for c in expected_features_orig if c in processed_orig.columns]
    orig_feature_names = common_orig
    print(f"Original features (aligned): {len(orig_feature_names)}")
    processed_orig[orig_feature_names] = processed_orig[orig_feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed_orig[orig_feature_names] = orig_scaler.transform(processed_orig[orig_feature_names])

    # ---- 加载模型 ----
    orig_cfg = {'sequence_length': 60, 'd_model': 256, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 512, 'dropout': 0.1}
    orig_model = OrigModel(len(orig_feature_names), orig_cfg, num_stocks)
    orig_model.load_state_dict(torch.load('./model/60_158+39/best_model.pth', map_location=DEVICE))
    orig_model.to(DEVICE).eval()

    opt_model = OptModel(len(opt_feature_names), opt_config, num_stocks)
    opt_model.load_state_dict(torch.load('./model/60_158+39+market/best_model.pth', map_location=DEVICE))
    opt_model.to(DEVICE).eval()
    print("Models loaded.")

    # ---- 逐日回测 ----
    results = {'opt': [], 'orig': [], 'market': []}

    for date in tqdm(test_dates, desc='Backtesting'):
        # 全市场等权
        mkt_rets = []
        for sid in stock_ids:
            fut = processed_orig[(processed_orig['股票代码'] == sid) & (processed_orig['日期'] > date)].sort_values('日期')
            if len(fut) >= 5:
                o1 = fut.iloc[0]['开盘']; c5 = fut.iloc[4]['收盘']
                if o1 > 1e-4: mkt_rets.append((c5-o1)/o1)
        if mkt_rets:
            results['market'].append(np.mean(mkt_rets))

        for label, model, processed, features, model_type in [
            ('opt', opt_model, processed_opt, opt_feature_names, 'opt'),
            ('orig', orig_model, processed_orig, orig_feature_names, 'orig'),
        ]:
            seqs, seq_ids, seq_idx = [], [], []
            for sid in stock_ids:
                hist = processed[(processed['股票代码'] == sid) & (processed['日期'] <= date)].sort_values('日期').tail(SEQUENCE_LENGTH)
                if len(hist) == SEQUENCE_LENGTH:
                    seqs.append(hist[features].values.astype(np.float32))
                    seq_ids.append(sid)
                    seq_idx.append(int(hist['instrument'].values[-1]))

            if len(seqs) < TOP_K:
                continue

            x = torch.from_numpy(np.stack(seqs)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if model_type == 'opt':
                    idx = torch.LongTensor(seq_idx).unsqueeze(0).to(DEVICE)
                    scores, _ = model(x, idx)
                else:
                    scores = model(x)
            scores = scores.squeeze(0).cpu().numpy()

            top_idx = np.argsort(scores)[::-1][:TOP_K]
            top_stocks = [seq_ids[i] for i in top_idx]

            rets = []
            for sid in top_stocks:
                fut = processed[(processed['股票代码'] == sid) & (processed['日期'] > date)].sort_values('日期')
                if len(fut) >= 5:
                    o1 = fut.iloc[0]['开盘']
                    c5 = fut.iloc[4]['收盘']
                    if o1 > 1e-4:
                        rets.append((c5 - o1) / o1)
            if len(rets) == TOP_K:
                results[label].append(np.mean(rets))

    # 只保留两个模型都有结果的日子做公平对比
    min_len = min(len(results['opt']), len(results['orig']))
    if min_len > 0:
        results['opt'] = results['opt'][:min_len]
        results['orig'] = results['orig'][:min_len]
        results['market'] = results['market'][:min_len]
    print(f"Valid days: opt={len(results['opt'])}, orig={len(results['orig'])}, market={len(results['market'])}")

    # ---- 计算指标 ----
    def calc_metrics(daily_rets):
        rets = np.array(daily_rets)
        n = len(rets)
        cum = np.prod(1 + rets) - 1
        ann = (1+cum)**(252/max(n,1)) - 1
        vol = np.std(rets, ddof=1)*np.sqrt(252)
        sharpe = (ann - 0.02)/(vol + 1e-12)
        cum_curve = np.cumprod(1+rets)
        peak = np.maximum.accumulate(cum_curve)
        mdd = np.min((cum_curve - peak)/peak)
        win = np.mean(rets > 0)
        return {'累计收益': cum, '年化收益': ann, '年化波动': vol, '夏普比率': sharpe, '最大回撤': mdd, '胜率': win, '日均收益': np.mean(rets)}

    print("\n" + "="*65)
    print("                     回 测 评 估 报 告")
    print("="*65)
    print(f"测试周期: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} 天)")
    print(f"模型: 优化={len(results['opt'])}天, 原始={len(results['orig'])}天")
    print("-"*65)
    print(f"{'指标':<16} {'优化模型':>14} {'原始模型':>14} {'沪深300等权':>14}")
    print("-"*65)

    for name, key in [('累计收益率','累计收益'),('年化收益率','年化收益'),('年化波动率','年化波动'),
                       ('夏普比率','夏普比率'),('最大回撤','最大回撤'),('胜率','胜率'),('日均收益','日均收益')]:
        vals = []
        for label in ['opt','orig','market']:
            if results[label]:
                m = calc_metrics(results[label])
                v = m[key]
                if '收益' in name or '回撤' in name: vals.append(f'{v:+.2%}')
                elif '率' in name: vals.append(f'{v:.2%}' if key=='胜率' else f'{v:.2%}')
                else: vals.append(f'{v:.3f}')
            else:
                vals.append('N/A')
        print(f"{name:<16} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    print("-"*65)
    # 换手率
    if len(results['opt']) > 1:
        # 简单估算：记录每相邻两天的选股变化
        print("\n(换手率统计需要每日持仓记录，此处省略)")

    # 超额收益对比
    if results['opt'] and results['orig']:
        opt_excess = np.array(results['opt']) - np.array(results['market'])
        orig_excess = np.array(results['orig']) - np.array(results['market'])
        print(f"\n{'超额收益(vs市场)':<16} {np.prod(1+opt_excess)-1:>+13.2%} {np.prod(1+orig_excess)-1:>+13.2%}")

    print("="*65)

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
