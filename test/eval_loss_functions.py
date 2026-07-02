"""
评估所有损失函数变体: WeightedRanking vs ListMLE vs ApproxNDCG vs LambdaRank vs Hybrid
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm

from config_v5 import SEQUENCE_LENGTH, MAX_STOCKS_PER_CHUNK, USE_AMP, MC_SAMPLES
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, _build_label_and_clean
from market_regime import compute_market_regime
from quality_filter import (
    bounce_confirm, compute_quality_score, volatility_filter, equal_weight_allocate
)

ROOT = os.path.join(os.path.dirname(__file__), '..')
TRAIN = os.path.join(ROOT, 'data', 'train.csv')
TEST = os.path.join(ROOT, 'data', 'test.csv')

LOSS_VARIANTS = {
    'Weighted': 'model/stock_emb_8_ensemble',
    'ListMLE': 'model/stock_emb_8_listmle',
    'ApproxNDCG': 'model/stock_emb_8_approxndcg',
    'LambdaRank': 'model/stock_emb_8_lambdarank',
    'Hybrid': 'model/stock_emb_8_hybrid',
}

TEST_CUTOFFS = {
    'Jun W1 (5/29)': {'cutoff': '2026-05-29', 't1_open': '2026-06-01', 't5_open': '2026-06-05'},
    'Jun W2 (6/5)': {'cutoff': '2026-06-05', 't1_open': '2026-06-08', 't5_open': '2026-06-12'},
}


def load_data():
    train = pd.read_csv(TRAIN, dtype={'股票代码': str})
    train['股票代码'] = train['股票代码'].str.zfill(6)
    train['日期'] = pd.to_datetime(train['日期'].str.replace(' 00:00:00', ''), format='mixed')
    test = pd.read_csv(TEST, dtype={'股票代码': str})
    test['股票代码'] = test['股票代码'].str.zfill(6)
    test['日期'] = pd.to_datetime(test['日期'].str.replace(' 00:00:00', ''), format='mixed')
    all_data = pd.concat([train, test], ignore_index=True)
    all_data = all_data.sort_values(['股票代码', '日期']).reset_index(drop=True)
    return all_data


def load_models(model_dir, feature_dim, num_stocks, device):
    with open(os.path.join(model_dir, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        if ec['type'] == 'transformer':
            m = StockTransformerExpert(feature_dim, ec, num_stocks)
        else:
            m = ConvStockExpert(feature_dim, ec, num_stocks)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        models.append(m)
    return models


def mc_predict(models, x_tensor, device):
    """MC Dropout 推理"""
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'
    n_models = len(models)

    seeds = [42, 142, 242, 342, 442]
    all_rounds = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rnd_scores = []
        for model in models:
            model.train()
            mc = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x_tensor.size(1) <= chunk_size:
                            s = model(x_tensor).squeeze(0)
                        else:
                            s = torch.cat([model(x_tensor[:, i:i+chunk_size]).squeeze(0)
                                         for i in range(0, x_tensor.size(1), chunk_size)], dim=0)
                    mc.append(s.cpu().numpy())
            rnd_scores.append(np.mean(mc, axis=0))
        fused = np.zeros_like(rnd_scores[0])
        for s in rnd_scores:
            fused += s / n_models
        all_rounds.append(fused)
    return np.mean(all_rounds, axis=0)


def process_week(data, models, model_dir, cutoff_dt, info, stock_ids, device):
    """对单个周进行评估"""
    winsor = json.load(open(os.path.join(model_dir, 'winsor_bounds.json')))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

    fc = feature_cloums_map['158+39']

    # Preprocess with this model's scaler
    df = data.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    p = pd.concat([engineer_features_158plus39(g) for g in tqdm(groups, desc='FE', leave=False)]).reset_index(drop=True)
    p['日期'] = pd.to_datetime(p['日期'].astype(str).str.replace(' 00:00:00', ''), format='mixed')
    stock_ids_actual = sorted(p['股票代码'].unique())
    sid2idx = {s: i for i, s in enumerate(stock_ids_actual)}
    p['instrument'] = p['股票代码'].map(sid2idx)

    for col, (lo, hi) in winsor.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    common = [c for c in scaler.feature_names_in_ if c in p.columns]
    p[common] = scaler.transform(p[common])

    # Build sequences
    seqs, sids = [], []
    for sid in stock_ids_actual:
        hist = p[(p['股票代码'] == sid) & (p['日期'] <= cutoff_dt)].sort_values('日期').tail(SEQUENCE_LENGTH)
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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    data = load_data()
    stock_ids_all = sorted(data['股票代码'].unique())

    # Load models for each variant
    all_models = {}
    for name, model_dir in LOSS_VARIANTS.items():
        cfg_path = os.path.join(ROOT, model_dir, 'ensemble_config.json')
        if not os.path.exists(cfg_path):
            print(f"  SKIP {name}: {cfg_path} not found")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        models = load_models(
            os.path.join(ROOT, model_dir),
            cfg.get('feature_dim', 197),
            cfg.get('num_stocks', 300),
            device)
        all_models[name] = {
            'models': models,
            'dir': os.path.join(ROOT, model_dir),
        }
        print(f"  {name}: {len(models)} experts loaded")

    results = {name: {} for name in all_models}

    for label_w, info in TEST_CUTOFFS.items():
        cutoff_dt = pd.to_datetime(info['cutoff'])
        print(f"\n--- {label_w} (cutoff={info['cutoff']}) ---")

        for name, model_info in all_models.items():
            result = process_week(data, model_info['models'], model_info['dir'],
                                 cutoff_dt, info, stock_ids_all, device)
            if result:
                results[name][label_w] = result
                skip = 'SKIP' if result.get('skip') else ''
                print(f"  {name:12s}: Top5={result['stocks']} | return={result['return']:+.4f} "
                      f"({result['return']*100:+.2f}%) {skip}")

    # Summary
    print(f"\n{'='*75}")
    print("LOSS FUNCTION COMPARISON: WeightedRanking vs ListMLE vs ApproxNDCG vs LambdaRank vs Hybrid")
    print(f"{'Loss':<14} | {'Jun W1':>10} | {'Jun W2':>10} | {'Average':>10} | {'vs Baseline':>10}")
    print("-" * 75)

    baseline_avg = None
    for name in ['Weighted', 'ListMLE', 'ApproxNDCG', 'LambdaRank', 'Hybrid']:
        if name not in results:
            continue
        w1_r = results[name].get('Jun W1 (5/29)', {}).get('return', 0) * 100
        w2_r = results[name].get('Jun W2 (6/5)', {}).get('return', 0) * 100
        avg = (w1_r + w2_r) / 2

        if name == 'Weighted':
            baseline_avg = avg
            diff_str = "—"
        elif baseline_avg is not None:
            diff_str = f"{avg - baseline_avg:+9.2f}%"
        else:
            diff_str = "N/A"

        print(f"{name:<14} | {w1_r:+9.2f}% | {w2_r:+9.2f}% | {avg:+9.2f}% | {diff_str:>10}")

    print("-" * 75)

    # Find best loss
    best_name, best_avg = None, -float('inf')
    for name, res in results.items():
        vals = [r.get('return', 0) for r in res.values()]
        if vals:
            avg = np.mean(vals) * 100
            if avg > best_avg:
                best_avg = avg
                best_name = name
    print(f"Best: {best_name} ({best_avg:+.2f}%)")


if __name__ == '__main__':
    main()
