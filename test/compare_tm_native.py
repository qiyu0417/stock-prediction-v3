"""
组员v1_ensemble原始管线: 完全复制predict_smart.py逻辑
MC 20 + MetaAggregator + σ仓位 + temperature 0.3 + 30% cap
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
import gc

from config_v5 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert, MonthSeasonalExpert, MetaAggregator
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map

TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
TM_DIR = "C:/Users/73065/Desktop/gupiao-3-model/gupiao-3-model/model/v1_ensemble"

MONTHS = {
    '2026-01': ('2025-12-31', ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']),
    '2026-02': ('2026-01-27', ['2026-02-02', '2026-02-03', '2026-02-04', '2026-02-05', '2026-02-06']),
    '2026-03': ('2026-02-27', ['2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06']),
    '2026-04': ('2026-03-31', ['2026-04-01', '2026-04-02', '2026-04-03', '2026-04-07', '2026-04-08']),
    '2026-05': ('2026-04-30', ['2026-05-04', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08']),
}


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_tm(df, stockid2idx, scaler):
    """完全复制他们的预处理"""
    fe = feature_engineer_func_map[FEATURE_NUM]
    fc = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    processed = pd.concat([fe(g) for g in tqdm(groups, desc='FE', leave=False)]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])
    processed[fc] = processed[fc].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    common = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[common] = scaler.transform(processed[common])
    return processed, common


def build_sequences(data, features, stock_ids, target_date):
    sequences, seq_stock_ids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= target_date)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            sequences.append(hist[features].values.astype(np.float32))
            seq_stock_ids.append(sid)
    return np.asarray(sequences, dtype=np.float32) if sequences else np.array([]), seq_stock_ids


def load_tm(fdim, nstocks, device):
    with open(os.path.join(TM_DIR, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    experts, meta = [], None
    for ec in cfg['expert_configs']:
        path = os.path.join(TM_DIR, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        t = ec.get('type','transformer')
        if t == 'transformer': m = StockTransformerExpert(fdim, ec, nstocks)
        elif t == 'conv': m = ConvStockExpert(fdim, ec, nstocks)
        elif t == 'month_seasonal': m = MonthSeasonalExpert(fdim, ec, nstocks)
        else: continue
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device); experts.append(m)
    meta_path = os.path.join(TM_DIR, 'meta_aggregator.pth')
    if os.path.exists(meta_path):
        meta = MetaAggregator(len(experts), nstocks, hidden_dim=64).to(device)
        meta.load_state_dict(torch.load(meta_path, map_location=device))
        meta.eval()
    return experts, meta


def tm_predict_native(experts, meta, x, device):
    """完全复制 predict_smart.py: MC 20 → Meta → σ权重 + 30%cap"""
    mc_samples = 20
    all_scores = []
    for e in experts:
        e.train()
        mc = []
        with torch.no_grad():
            for _ in range(mc_samples):
                mc.append(e(x).squeeze(0))
        all_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())

    # MetaAggregator
    es = torch.from_numpy(np.stack(all_scores, axis=-1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        final = meta(es).squeeze(0).cpu().numpy()

    return final


def tm_smart_weights(final_scores, top_n=5):
    """predict_smart.py 的权重算法"""
    order = np.argsort(final_scores)[::-1]
    top_scores = final_scores[order[:top_n]]

    # Softmax t=0.3
    temperature = 0.3
    weights = np.exp(top_scores / temperature)
    weights = weights / weights.sum()

    # 30% cap
    MAX_SINGLE = 0.30
    for _ in range(10):
        overflow = 0.0
        capped = 0
        for i in range(top_n):
            if weights[i] > MAX_SINGLE:
                overflow += weights[i] - MAX_SINGLE
                weights[i] = MAX_SINGLE
                capped += 1
        if overflow > 0 and capped < top_n:
            uc = top_n - capped
            for i in range(top_n):
                if weights[i] < MAX_SINGLE:
                    weights[i] += overflow / uc
        else:
            break

    # σ仓位
    max_score = top_scores[0]
    mean_score = final_scores.mean()
    std_score = final_scores.std()
    confidence = (max_score - mean_score) / (std_score + 1e-8)

    if confidence < 1.0:
        pos_ratio = 0.30
    elif confidence < 2.0:
        pos_ratio = 0.50 + (confidence - 1.0) * 0.25
    else:
        pos_ratio = min(1.0, 0.75 + (confidence - 2.0) * 0.10)

    final_weights = weights * pos_ratio
    return order, final_weights, confidence, pos_ratio


def calc_ret(stocks, wts, data, dates):
    wd = data[data['日期'].isin(pd.to_datetime(dates))]
    f = wd[wd['股票代码'].isin(stocks)]
    if f.empty: return 0.0
    total = 0.0
    for sid, w in zip(stocks, wts):
        sw = f[f['股票代码']==sid].sort_values('日期')
        if len(sw)>=2: total += w*(sw.iloc[-1]['开盘']-sw.iloc[0]['开盘'])/sw.iloc[0]['开盘']
    return total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    full_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码':str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    all_stocks = sorted(full_df['股票代码'].unique())
    fdim = len(feature_cloums_map[FEATURE_NUM])
    nstocks = len(all_stocks)

    tm_scaler = joblib.load(os.path.join(TM_DIR, 'scaler.pkl'))
    tm_experts, tm_meta = load_tm(fdim, nstocks, device)
    print(f"TM: {len(tm_experts)} experts + Meta\n")

    results = []
    for month, (cutoff, week_dates) in MONTHS.items():
        print(f"{'='*60}")
        print(f"  {month} | cutoff: {cutoff}")
        train_df = full_df[full_df['日期']<=cutoff].copy()
        sid2idx = {s:i for i,s in enumerate(all_stocks)}

        ptm, ctm = preprocess_tm(train_df, sid2idx, tm_scaler)
        stm, ids = build_sequences(ptm, ctm, all_stocks, pd.to_datetime(cutoff))
        xtm = torch.from_numpy(stm).unsqueeze(0).to(device)

        final = tm_predict_native(tm_experts, tm_meta, xtm, device)
        order, wts, conf, pos = tm_smart_weights(final)

        top5 = [ids[i] for i in order[:5] if i < len(ids)]
        ret = calc_ret(top5, wts, full_df, week_dates)

        # Also raw top5 equal weight
        raw_top = [ids[i] for i in order[:5] if i < len(ids)]
        ret_eq = calc_ret(raw_top, [0.2]*len(raw_top), full_df, week_dates)

        print(f"  σ={conf:.2f}, pos={pos:.0%}")
        print(f"  TM等权: {raw_top} → {ret_eq:+.4%}")
        print(f"  TM智能: {list(zip(top5, [f'{w:.1%}' for w in wts[:5]]))} → {ret:+.4%}\n")

        results.append({'m':month, 'eq':ret_eq, 'smart':ret, 'conf':conf, 'pos':pos})
        del train_df, ptm; gc.collect()

    print("="*60)
    print(f"  {'Month':<8} {'TM等权':>8} {'TM智能':>8} {'σ':>6} {'仓位':>6}")
    teq=tsm=0
    for r in results:
        teq+=r['eq']; tsm+=r['smart']
        print(f"  {r['m']:<8} {r['eq']:>+8.4%} {r['smart']:>+8.4%} {r['conf']:>5.2f} {r['pos']:>5.0%}")
    print(f"  {'累计':<8} {teq:>+8.4%} {tsm:>+8.4%}")


if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    main()
