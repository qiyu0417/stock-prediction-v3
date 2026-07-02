"""
Stock ID Embedding vs V7 对比评估: 计算六月前两周Top5收益
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
import gc

from config_v5 import SEQUENCE_LENGTH, MAX_STOCKS_PER_CHUNK, USE_AMP, MC_SAMPLES
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39, add_industry_features, INDUSTRY_COLS
from train import feature_cloums_map, _build_label_and_clean
from market_regime import compute_market_regime
from quality_filter import (
    bounce_confirm, compute_quality_score, volatility_filter, equal_weight_allocate
)

ROOT = os.path.join(os.path.dirname(__file__), '..')
TRAIN = os.path.join(ROOT, 'data', 'train.csv')
TEST = os.path.join(ROOT, 'data', 'test.csv')
V7_DIR = os.path.join(ROOT, 'model', 'v7_ensemble')
IND_DIR = os.path.join(ROOT, 'model', 'stock_emb_8_ensemble')

# 测试日期: T+1开盘 ~ T+5开盘
TEST_CUTOFFS = {
    'Jun W1 (5/29)': {
        'cutoff': '2026-05-29',
        't1_open': '2026-06-01',
        't5_open': '2026-06-05',
    },
    'Jun W2 (6/5)': {
        'cutoff': '2026-06-05',
        't1_open': '2026-06-08',
        't5_open': '2026-06-12',
    },
}


def load_data(add_industry=False):
    train = pd.read_csv(TRAIN, dtype={'股票代码': str})
    train['股票代码'] = train['股票代码'].str.zfill(6)
    train['日期'] = pd.to_datetime(train['日期'].str.replace(' 00:00:00', ''), format='mixed')

    test = pd.read_csv(TEST, dtype={'股票代码': str})
    test['股票代码'] = test['股票代码'].str.zfill(6)
    test['日期'] = pd.to_datetime(test['日期'].str.replace(' 00:00:00', ''), format='mixed')

    all_data = pd.concat([train, test], ignore_index=True)
    all_data = all_data.sort_values(['股票代码', '日期']).reset_index(drop=True)

    if add_industry:
        all_data = add_industry_features(all_data)

    return all_data


def prepare_features(df, winsor, scaler, add_industry=False):
    """特征工程 + Winsorization + 标准化"""
    from train import _engineer_158plus39_industry

    if add_industry:
        fe = _engineer_158plus39_industry
        fc = feature_cloums_map['158+39+industry']
    else:
        def _fe(df):
            return engineer_features_158plus39(df, add_market=False)
        fe = _fe
        fc = feature_cloums_map['158+39']

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    p = pd.concat([fe(g) for g in tqdm(groups, desc='FE', leave=False)]).reset_index(drop=True)
    p['日期'] = pd.to_datetime(p['日期'].astype(str).str.replace(' 00:00:00', ''), format='mixed')

    # add instrument (stock index) - required by scaler
    stock_ids = sorted(p['股票代码'].unique())
    sid2idx = {s: i for i, s in enumerate(stock_ids)}
    p['instrument'] = p['股票代码'].map(sid2idx)

    # winsor
    for col, (lo, hi) in winsor.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    # scale
    common = [c for c in scaler.feature_names_in_ if c in p.columns]
    p[common] = scaler.transform(p[common])
    return p, common


def load_models(model_dir, feature_dim, num_stocks, device):
    with open(os.path.join(model_dir, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        try:
            if ec['type'] == 'transformer':
                m = StockTransformerExpert(feature_dim, ec, num_stocks)
            else:
                m = ConvStockExpert(feature_dim, ec, num_stocks)
            m.load_state_dict(torch.load(path, map_location=device))
            m.to(device)
            models.append(m)
        except Exception as e:
            print(f"  skip {ec['name']}: {e}")
    return models, [1.0 / len(models)] * len(models)


def mc_predict(models, weights, x, device):
    """MC Dropout 推理"""
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'
    all_fused = []
    for r in range(5):
        torch.manual_seed(42 + r * 100)
        np.random.seed(42 + r * 100)
        rnd_scores = []
        for model in models:
            model.train()
            mc = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x.size(1) <= chunk_size:
                            s = model(x).squeeze(0)
                        else:
                            s = torch.cat([model(x[:, i:i+chunk_size]).squeeze(0)
                                          for i in range(0, x.size(1), chunk_size)], dim=0)
                    mc.append(s.cpu().numpy())
            rnd_scores.append(np.mean(mc, axis=0))
        # fuse experts
        fused = np.zeros_like(rnd_scores[0])
        for w, s in zip(weights, rnd_scores):
            fused += w * s
        all_fused.append(fused)
    return np.mean(all_fused, axis=0)


def evaluate(model_dir, label, add_industry, device):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Model: {model_dir}")

    data = load_data(add_industry=add_industry)
    winsor = json.load(open(os.path.join(model_dir, 'winsor_bounds.json')))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

    pdata, features = prepare_features(data, winsor, scaler, add_industry=add_industry)

    with open(os.path.join(model_dir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    fdim = cfg.get('feature_dim', len(features))
    nstocks = cfg.get('num_stocks', 300)

    models, weights = load_models(model_dir, fdim, nstocks, device)
    print(f"  loaded {len(models)} experts")

    stock_ids = sorted(data['股票代码'].unique())
    results = {}

    for label_w, info in TEST_CUTOFFS.items():
        cutoff = info['cutoff']
        cutoff_dt = pd.to_datetime(cutoff)

        # build input batch
        seqs, sids = [], []
        for sid in stock_ids:
            hist = pdata[(pdata['股票代码'] == sid) & (pdata['日期'] <= cutoff_dt)].sort_values('日期').tail(SEQUENCE_LENGTH)
            if len(hist) == SEQUENCE_LENGTH:
                seqs.append(hist[features].values.astype(np.float32))
                sids.append(sid)

        if not seqs:
            print(f"  {label_w}: no sequences!")
            continue
        x = torch.FloatTensor(np.stack(seqs)).unsqueeze(0).to(device)

        raw = mc_predict(models, weights, x, device)
        raw_scores = {sid: float(raw[i]) for i, sid in enumerate(sids)}

        # Market regime
        regime = compute_market_regime(data, features, sids, cutoff_dt)
        skip = regime.get('skip_trading', False)

        if skip:
            print(f"  {label_w}: SKIP (market regime)")
            results[label_w] = {'return': 0, 'skip': True, 'stocks': []}
            continue

        # Post-processing
        # volatility filter
        kept = volatility_filter(data, sids, cutoff_dt, top_pct=0.95)
        # bounce confirm
        confirmed = bounce_confirm(data, kept, cutoff_dt, threshold=0.008)
        # quality
        quality = compute_quality_score(data, kept, cutoff_dt)

        # adjust scores
        adjusted = {}
        for sid in kept:
            s = raw_scores.get(sid, 0)
            if sid not in confirmed:
                s *= 0.92
            q = quality.get(sid, 0.5)
            s += (q - 0.5) * 0.05
            adjusted[sid] = s

        # select top 5
        ranked = sorted(adjusted.items(), key=lambda x: -x[1])
        selected, weights_alloc = equal_weight_allocate([s for s, _ in ranked], 5)

        # Calculate actual return
        ret = 0
        if selected:
            t1_map = {}
            t5_map = {}
            for sid in selected:
                sd = data[(data['股票代码'] == sid) & (data['日期'].dt.strftime('%Y-%m-%d') == info['t1_open'])]
                if len(sd) > 0:
                    t1_map[sid] = float(sd['开盘'].iloc[0])
                sd5 = data[(data['股票代码'] == sid) & (data['日期'].dt.strftime('%Y-%m-%d') == info['t5_open'])]
                if len(sd5) > 0:
                    t5_map[sid] = float(sd5['开盘'].iloc[0])

            stock_rets = []
            for sid in selected:
                if sid in t1_map and sid in t5_map and t1_map[sid] > 0:
                    r = (t5_map[sid] - t1_map[sid]) / t1_map[sid]
                    stock_rets.append(r)
            if stock_rets:
                ret = np.mean(stock_rets)

        print(f"  {label_w}: Top5={selected} | return={ret:+.4f} ({ret*100:+.2f}%)")
        results[label_w] = {'return': ret, 'skip': False, 'stocks': selected}

        del x; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # V7 baseline
    r7 = evaluate(V7_DIR, "V7 (DeepSleep V1)", add_industry=False, device=device)

    # Stock Emb dim=8edding
    ri = evaluate(IND_DIR, "Stock Emb dim=8", add_industry=False, device=device)

    # Comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'Week':<18} | {'V7':>10} | {'Stock Emb dim=8':>12} | {'Diff':>10}")
    print("-" * 58)
    for label_w in TEST_CUTOFFS:
        v7_r = r7.get(label_w, {}).get('return', 0) * 100
        ind_r = ri.get(label_w, {}).get('return', 0) * 100
        diff = ind_r - v7_r
        print(f"{label_w:<18} | {v7_r:+9.2f}% | {ind_r:+11.2f}% | {diff:+9.2f}%")

    avg_v7 = np.mean([r.get('return', 0) for r in r7.values()]) * 100
    avg_ind = np.mean([r.get('return', 0) for r in ri.values()]) * 100
    print("-" * 58)
    print(f"{'Average':<18} | {avg_v7:+9.2f}% | {avg_ind:+11.2f}% | {avg_ind-avg_v7:+9.2f}%")


if __name__ == '__main__':
    main()
