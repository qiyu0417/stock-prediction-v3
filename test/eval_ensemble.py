"""Multi-model ensemble evaluation: any combination of trained models"""
import os, sys, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC_PASSES = 5
SEQUENCE_LENGTH = 60

MODELS = {
    'Hybrid dim=8':    {'dir': 'model/stock_emb_8_hybrid',          'glu': False},
    'k=3 T=0.5':       {'dir': 'model/stock_emb_8_listmle_k3_t0.5', 'glu': False},
    'RankGLU':         {'dir': 'model/stock_emb_8_rankglu',         'glu': True},
    'EMA':             {'dir': 'model/stock_emb_8_ema',             'glu': False},
    'Multitask':       {'dir': 'model/stock_emb_8_multitask',       'glu': False},
}

ENSEMBLES = [
    # 2-model new combos
    ['EMA', 'Multitask'],
    ['EMA', 'Hybrid dim=8'],
    ['EMA', 'k=3 T=0.5'],
    ['Multitask', 'Hybrid dim=8'],
    ['Multitask', 'k=3 T=0.5'],
    # 3-model
    ['EMA', 'Hybrid dim=8', 'k=3 T=0.5'],
    ['EMA', 'Multitask', 'Hybrid dim=8'],
    ['Multitask', 'Hybrid dim=8', 'k=3 T=0.5'],
    ['EMA', 'Multitask', 'k=3 T=0.5'],
]


def preprocess_eval(df, stockid2idx, scaler, winsor_bounds):
    from config_stock_emb_8 import FEATURE_NUM
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    min_rows = SEQUENCE_LENGTH + 10
    groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= min_rows]
    if len(groups) == 0:
        return None, None
    processed = pd.concat([feature_engineer(g) for g in groups]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=True)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=feature_columns)
    for col, (lo, hi) in winsor_bounds.items():
        if col in processed.columns:
            processed[col] = processed[col].clip(lo, hi)
    processed[feature_columns] = scaler.transform(processed[feature_columns])
    return processed, feature_columns


def load_experts_for_model(model_name, feature_dim, num_stocks, device, _orig_fa):
    info = MODELS[model_name]
    model_dir = info['dir']
    needs_glu = info['glu']

    import ensemble_models as _em
    if needs_glu:
        class _GatedFeatureAttention(nn.Module):
            def __init__(self, d_model, dropout=0.1):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.Linear(d_model, d_model // 2), nn.Tanh(),
                    nn.Linear(d_model // 2, 1), nn.Softmax(dim=1))
                self.gate = nn.Sequential(
                    nn.Linear(d_model, d_model // 2), nn.ReLU(),
                    nn.Linear(d_model // 2, d_model), nn.Sigmoid())
                self.dropout = nn.Dropout(dropout)
            def forward(self, x):
                attn_weights = self.attention(x)
                attended = torch.sum(x * attn_weights, dim=1)
                return self.dropout(attended * self.gate(attended))
        _em.FeatureAttention = _GatedFeatureAttention
    else:
        _em.FeatureAttention = _orig_fa  # restore original

    from ensemble_models import StockTransformerExpert, ConvStockExpert

    with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
        cfg = json.load(f)
    embed_dim = cfg.get('stock_embed_dim', 8)
    expert_configs = cfg['expert_configs']
    models = []
    for ec in expert_configs:
        path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        ec_copy = dict(ec)
        ec_copy['stock_embed_dim'] = embed_dim
        if ec['type'] == 'transformer':
            model = StockTransformerExpert(feature_dim, ec_copy, num_stocks)
        else:
            model = ConvStockExpert(feature_dim, ec_copy, num_stocks)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.train()
        models.append(model)
    return models


def ensure_dir_ready(model_name):
    info = MODELS[model_name]
    model_dir = info['dir']
    if not os.path.exists(model_dir):
        return False
    config_path = os.path.join(model_dir, 'ensemble_config.json')
    if not os.path.exists(config_path):
        return False
    return True


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")
    set_seed(42)

    available = {name: ensure_dir_ready(name) for name in MODELS}
    for name, ok in available.items():
        print(f"  {name}: {'READY' if ok else 'MISSING'}")

    train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
    train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6)
    train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')
    test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
    test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
    test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['股票代码', '日期'], keep='last')
    test_dates = sorted(test_df['日期'].unique())
    raw_data = full_df.copy()
    all_stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_stock_ids)}

    base_model = 'Hybrid dim=8'
    info = MODELS[base_model]
    model_dir = info['dir']
    with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
        cfg = json.load(f)
    feature_dim = cfg['feature_dim']
    num_stocks = cfg['num_stocks']

    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    with open(os.path.join(model_dir, 'winsor_bounds.json'), 'r') as f:
        winsor_bounds = json.load(f)

    processed, features = preprocess_eval(full_df, stockid2idx, scaler, winsor_bounds)
    n_feats = len(features)

    scorers = {}
    import ensemble_models as _em
    _orig_fa = _em.FeatureAttention  # save once, before any patching
    for name in MODELS:
        if not available[name]:
            continue
        print(f'\nLoading: {name}...')
        models = load_experts_for_model(name, feature_dim, num_stocks, device, _orig_fa)
        print(f'  {len(models)} experts loaded')
        scorers[name] = models

    pred_dates = ['2026-06-01', '2026-06-08']
    results = {}

    for ensemble_names in ENSEMBLES:
        label = ' + '.join(ensemble_names)
        if not all(available.get(n) for n in ensemble_names):
            print(f'\nSKIP {label}: missing')
            continue

        print(f'\n=== {label} ===')
        returns_list = []

        for pred_date_str in pred_dates:
            pred_dt = pd.to_datetime(pred_date_str)
            hist = processed[processed['日期'] <= pred_date_str]
            avail_stocks = hist['股票代码'].unique()
            stock_ids = sorted(avail_stocks)
            n_stocks = len(stock_ids)
            if n_stocks < 5:
                continue

            seq_len = SEQUENCE_LENGTH
            sequences = np.zeros((1, n_stocks, seq_len, n_feats), dtype=np.float32)
            valid_mask = np.zeros(n_stocks, dtype=bool)
            for i, sid in enumerate(stock_ids):
                stock_data = hist[hist['股票代码'] == sid].sort_values('日期')
                if len(stock_data) >= seq_len:
                    sequences[0, i] = stock_data[features].values[-seq_len:].astype(np.float32)
                    valid_mask[i] = True

            seq_t = torch.FloatTensor(sequences).to(device)

            all_model_scores = []
            for mname in ensemble_names:
                models = scorers[mname]
                if not models:
                    break
                mc_scores_list = []
                for _ in range(MC_PASSES):
                    pass_scores = []
                    for model in models:
                        with torch.no_grad():
                            pred = model(seq_t)
                            if isinstance(pred, tuple):
                                pred = pred[0]
                            pred = pred[0].cpu().numpy()
                        pass_scores.append(pred)
                    mc_scores_list.append(np.mean(pass_scores, axis=0))
                model_avg = np.mean(mc_scores_list, axis=0)
                all_model_scores.append(model_avg)

            if not all_model_scores:
                print(f'  SKIP {pred_date_str}: no model scores')
                continue
            ensemble_scores = np.mean(all_model_scores, axis=0)

            raw_scores = {}
            for i, sid in enumerate(stock_ids):
                raw_scores[sid] = float(ensemble_scores[i]) if valid_mask[i] else -float('inf')

            data = raw_data[raw_data['日期'] <= pred_date_str]
            filtered_ids = volatility_filter(data, stock_ids, pred_date_str, top_pct=0.95)
            bounce_flags = bounce_confirm(data, filtered_ids, pred_date_str)
            quality_scores = compute_quality_score(data, filtered_ids, pred_date_str)

            final_scores = {}
            for sid in filtered_ids:
                score = raw_scores.get(sid, -float('inf'))
                if sid not in bounce_flags:
                    score *= 0.92
                quality = quality_scores.get(sid, 0.5)
                score += (quality - 0.5) * 0.05
                final_scores[sid] = score

            ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            top5_ids = [sid for sid, _ in ranked[:5]]
            selected, weights = equal_weight_allocate(top5_ids)
            picks = list(zip(selected, weights))

            t1_date = None
            for d in test_dates:
                if d >= pd.to_datetime(pred_date_str):
                    t1_date = d
                    break
            t5_dates_in_range = [d for d in test_dates if d >= pd.to_datetime(pred_date_str)]
            t5_idx = min(4, len(t5_dates_in_range) - 1)
            t5_date = t5_dates_in_range[t5_idx] if t5_dates_in_range else test_dates[-1]

            rets = []
            for sid, weight in picks:
                t1_data = test_df[(test_df['股票代码'] == sid) & (test_df['日期'] == t1_date)]
                t5_data = test_df[(test_df['股票代码'] == sid) & (test_df['日期'] == t5_date)]
                if len(t1_data) == 0 or len(t5_data) == 0:
                    ret = 0.0
                else:
                    ret = (float(t5_data.iloc[0]['开盘']) - float(t1_data.iloc[0]['开盘'])) / float(t1_data.iloc[0]['开盘'])
                rets.append(ret * weight)

            week_ret = sum(rets)
            sids = [p[0] for p in picks]
            returns_list.append(week_ret)
            print(f'  {pred_date_str}: {sids} | {week_ret*100:+.2f}%')

        avg = np.mean(returns_list)
        w1 = returns_list[0] * 100 if len(returns_list) > 0 else 0
        w2 = returns_list[1] * 100 if len(returns_list) > 1 else 0
        results[label] = {'w1': w1, 'w2': w2, 'avg': avg * 100}
        print(f'  => W1: {w1:+.2f}%  W2: {w2:+.2f}%  Avg: {avg*100:+.2f}%')

    print(f'\n{"="*70}')
    print(f'{"Ensemble":<40} {"Jun W1":>10} {"Jun W2":>10} {"Avg":>10}')
    print('-' * 70)
    ranked = sorted(results.items(), key=lambda x: x[1]['avg'], reverse=True)
    for name, r in ranked:
        print(f'{name:<40} {r["w1"]:>+9.2f}% {r["w2"]:>+9.2f}% {r["avg"]:>+9.2f}%')
    print('Done!')


if __name__ == '__main__':
    main()
