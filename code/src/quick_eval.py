"""
快速收益评估：加载优化模型，逐日预测Top5，计算实际5日收益
"""
import sys, os, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'code', 'src'))


def _feat_fn(g):
    from utils import engineer_features_158plus39
    return engineer_features_158plus39(g, add_market=True)


def main():
    import numpy as np, pandas as pd, torch, joblib
    import multiprocessing as mp
    from utils import engineer_features_158plus39
    from model import StockTransformer

    SEQ_LEN, TOP_K = 60, 5

    raw = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'), dtype={'股票代码': str})
    raw['股票代码'] = raw['股票代码'].astype(str).str.zfill(6)
    raw['日期'] = pd.to_datetime(raw['日期'])
    stock_ids = sorted(raw['股票代码'].unique())
    sid2idx = {s: i for i, s in enumerate(stock_ids)}

    print("特征工程...")
    groups = [g for _, g in raw.groupby('股票代码', sort=False)]
    with mp.Pool(min(10, mp.cpu_count())) as pool:
        processed = pd.concat(list(pool.imap(_feat_fn, groups))).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(sid2idx)
    processed = processed.dropna(subset=['instrument'])
    processed['instrument'] = processed['instrument'].astype(np.int64)

    scaler = joblib.load(os.path.join(ROOT, 'model', '60_158+39+market', 'scaler.pkl'))
    features = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[features] = processed[features].fillna(0.0)
    processed[features] = scaler.transform(processed[features])
    print(f"特征维度: {len(features)}")

    cfg = {'sequence_length': 60, 'd_model': 256, 'nhead': 4, 'num_layers': 3,
           'dim_feedforward': 512, 'dropout': 0.1}
    model = StockTransformer(len(features), cfg, len(stock_ids))
    model.load_state_dict(torch.load(
        os.path.join(ROOT, 'model', '60_158+39+market', 'best_model.pth'),
        map_location='cpu'))
    model.eval()
    print("模型加载完成")

    all_dates = sorted(processed['日期'].unique())
    # 仅使用训练截止后的日期（2026-01-06起），保证完全样本外
    test_start = pd.Timestamp('2026-01-06')
    test_dates = [d for d in all_dates
                  if d >= test_start and d <= all_dates[-6]]
    print(f"测试日期: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} 天)")

    daily_rets = []
    for date in test_dates:
        seqs, ids, idxs = [], [], []
        for sid in stock_ids:
            hist = processed[(processed['股票代码'] == sid) & (processed['日期'] <= date)]
            hist = hist.sort_values('日期').tail(SEQ_LEN)
            if len(hist) == SEQ_LEN:
                seqs.append(hist[features].values.astype(np.float32))
                ids.append(sid)
                idxs.append(int(hist['instrument'].values[-1]))
        if len(seqs) < TOP_K:
            continue

        x = torch.from_numpy(np.stack(seqs)).unsqueeze(0)
        idx = torch.LongTensor(idxs).unsqueeze(0)
        with torch.no_grad():
            scores, _ = model(x, idx)
        scores = scores.squeeze(0).numpy()
        top = [ids[i] for i in np.argsort(scores)[::-1][:TOP_K]]

        rets = []
        for sid in top:
            fut = processed[(processed['股票代码'] == sid) & (processed['日期'] > date)]
            fut = fut.sort_values('日期')
            if len(fut) >= 5:
                r = (fut.iloc[4]['收盘'] - fut.iloc[0]['开盘']) / fut.iloc[0]['开盘']
                rets.append(r)
        if len(rets) == TOP_K:
            daily_rets.append(np.mean(rets))

    n = len(daily_rets)
    print(f"\n有效交易日: {n}")
    if n == 0:
        print("无有效数据")
        return

    r = np.array(daily_rets)
    cum = np.prod(1 + r) - 1
    ann = (1 + cum) ** (252 / n) - 1
    vol = np.std(r, ddof=1) * np.sqrt(252)
    sharpe = (ann - 0.02) / (vol + 1e-12)
    peak = np.maximum.accumulate(np.cumprod(1 + r))
    mdd = np.min((np.cumprod(1 + r) - peak) / peak)
    win = np.mean(r > 0)

    print(f"累计收益: {cum:+.2%}")
    print(f"年化收益: {ann:+.2%}")
    print(f"年化波动: {vol:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {mdd:.2%}")
    print(f"胜率: {win:.1%}")
    print(f"日均收益: {np.mean(r):+.4%}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
