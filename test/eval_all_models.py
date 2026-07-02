"""
Evaluate all model variants on Jun W1 + Jun W2 (2026).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm

from config_v5 import SEQUENCE_LENGTH, MAX_STOCKS_PER_CHUNK, USE_AMP, MC_SAMPLES
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map
from market_regime import compute_market_regime
from quality_filter import bounce_confirm, compute_quality_score, volatility_filter, equal_weight_allocate

ROOT = os.path.join(os.path.dirname(__file__), '..')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TEST_CUTOFFS = {
    'Jun W1': {'cutoff': '2026-05-29', 't1_open': '2026-06-01', 't5_open': '2026-06-05'},
    'Jun W2': {'cutoff': '2026-06-05', 't1_open': '2026-06-08', 't5_open': '2026-06-12'},
}

MODELS_TO_EVAL = {
    'v7 (baseline)': 'model/v7_ensemble',
    'emb4_wr': 'model/stock_emb_ensemble',
    'emb8_wr': 'model/stock_emb_8_ensemble',
    'emb8_approxndcg': 'model/stock_emb_8_approxndcg',
    'emb8_listmle_k10_t0.5': 'model/stock_emb_8_listmle_k10_t0.5',
    'emb8_listmle_k3_t1.0': 'model/stock_emb_8_listmle_k3_t1.0',
}


def load_data():
    train = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'), dtype={'股票代码': str})
    train['股票代码'] = train['股票代码'].str.zfill(6)
    train['日期'] = pd.to_datetime(train['日期'].str.replace(' 00:00:00', ''), format='mixed')
    test = pd.read_csv(os.path.join(ROOT, 'data', 'test.csv'), dtype={'股票代码': str})
    test['股票代码'] = test['股票代码'].str.zfill(6)
    test['日期'] = pd.to_datetime(test['日期'].str.replace(' 00:00:00', ''), format='mixed')
    all_data = pd.concat([train, test], ignore_index=True)
    return all_data.sort_values(['股票代码', '日期']).reset_index(drop=True)


def load_models(model_dir, device):
    cfg_path = os.path.join(ROOT, model_dir, 'ensemble_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(ROOT, model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        if ec['type'] == 'transformer':
            m = StockTransformerExpert(cfg.get('feature_dim', 197), ec, cfg.get('num_stocks', 300))
        else:
            m = ConvStockExpert(cfg.get('feature_dim', 197), ec, cfg.get('num_stocks', 300))
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        models.append((ec['name'], m))
    return models


def preprocess(data, model_dir):
    winsor = json.load(open(os.path.join(ROOT, model_dir, 'winsor_bounds.json')))
    scaler = joblib.load(os.path.join(ROOT, model_dir, 'scaler.pkl'))
    fc = feature_cloums_map['158+39']

    df = data.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    p = pd.concat([engineer_features_158plus39(g) for g in tqdm(groups, desc='FE')]).reset_index(drop=True)
    p['日期'] = pd.to_datetime(p['日期'].astype(str).str.replace(' 00:00:00', ''), format='mixed')
    stock_ids = sorted(p['股票代码'].unique())
    sid2idx = {s: i for i, s in enumerate(stock_ids)}
    p['instrument'] = p['股票代码'].map(sid2idx)

    for col, (lo, hi) in winsor.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    common = [c for c in scaler.feature_names_in_ if c in p.columns]
    p[common] = scaler.transform(p[common])
    return p, common


def mc_predict(models, x, device):
    n_models = len(models)
    all_rounds = []
    for seed in [42, 142, 242, 342, 442]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rnd = []
        for _, model in models:
            model.train()
            mc = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    s = model(x).squeeze(0)
                    mc.append(s.cpu().numpy())
            rnd.append(np.mean(mc, axis=0))
        fused = np.zeros_like(rnd[0])
        for s in rnd:
            fused += s / n_models
        all_rounds.append(fused)
    return np.mean(all_rounds, axis=0)


def evaluate_week(data, pdata, common, models, cutoff_dt, info, device, label_w):
    stock_ids_all = sorted(pdata['股票代码'].unique())
    seqs, sids = [], []
    for sid in stock_ids_all:
        hist = pdata[(pdata['股票代码'] == sid) & (pdata['日期'] <= cutoff_dt)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            seqs.append(hist[common].values.astype(np.float32))
            sids.append(sid)

    if not seqs:
        return None

    x = torch.FloatTensor(np.stack(seqs)).unsqueeze(0).to(device)
    raw = mc_predict(models, x, device)
    raw_scores = {sid: float(raw[i]) for i, sid in enumerate(sids)}

    regime = compute_market_regime(data, common, sids, cutoff_dt)
    if regime.get('skip_trading', False):
        return {'return': 0, 'skip': True, 'stocks': []}

    kept = volatility_filter(data, sids, cutoff_dt, top_pct=0.95)
    confirmed = bounce_confirm(data, kept, cutoff_dt, threshold=0.008)
    quality = compute_quality_score(data, kept, cutoff_dt)

    adjusted = {}
    for sid in kept:
        s = raw_scores.get(sid, 0)
        if sid not in confirmed:
            s *= 0.92
        q = quality.get(sid, 0.5)
        s += (q - 0.5) * 0.05
        adjusted[sid] = s

    ranked = sorted(adjusted.items(), key=lambda x: -x[1])
    selected, _ = equal_weight_allocate([s for s, _ in ranked], 5)

    ret = 0
    if selected:
        t1_map, t5_map = {}, {}
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
                stock_rets.append((t5_map[sid] - t1_map[sid]) / t1_map[sid])
        if stock_rets:
            ret = np.mean(stock_rets)

    return {'return': ret, 'skip': False, 'stocks': selected}


def main():
    print(f"Device: {DEVICE}")
    data = load_data()

    results = {}
    for label, model_dir in MODELS_TO_EVAL.items():
        cfg_path = os.path.join(ROOT, model_dir, 'ensemble_config.json')
        if not os.path.exists(cfg_path):
            print(f"{label}: SKIP (no config)")
            continue

        models = load_models(model_dir, DEVICE)
        print(f"\n{label}: {len(models)} experts loaded")
        pdata, common = preprocess(data, model_dir)

        res = {}
        for label_w, info in TEST_CUTOFFS.items():
            cutoff_dt = pd.to_datetime(info['cutoff'])
            r = evaluate_week(data, pdata, common, models, cutoff_dt, info, DEVICE, label_w)
            if r:
                res[label_w] = r
                skip_str = ' SKIP' if r.get('skip') else ''
                print(f"  {label_w}: Top5={r['stocks']} | return={r['return']:+.4f} ({r['return']*100:+.2f}%){skip_str}")
        results[label] = res

    # Summary
    print()
    print('=' * 80)
    print(f"{'Model':<28} | {'Jun W1':>10} | {'Jun W2':>10} | {'Average':>10}")
    print('-' * 80)
    best_avg, best_name = -float('inf'), ''
    for label in MODELS_TO_EVAL:
        if label not in results:
            continue
        w1 = results[label].get('Jun W1', {}).get('return', 0) * 100
        w2 = results[label].get('Jun W2', {}).get('return', 0) * 100
        avg = (w1 + w2) / 2
        marker = ' <<' if avg > best_avg else ''
        print(f"{label:<28} | {w1:+9.2f}% | {w2:+9.2f}% | {avg:+9.2f}%{marker}")
        if avg > best_avg:
            best_avg = avg
            best_name = label
    print('-' * 80)
    print(f"Best: {best_name} ({best_avg:+.2f}%)")


if __name__ == '__main__':
    main()
