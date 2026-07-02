"""Comprehensive evaluation: all trained models + all 2-model ensembles"""
import os, sys, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC_PASSES = 5
SEQUENCE_LENGTH = 60

MODELS = {
    'Hybrid dim=8':    {'dir': 'model/stock_emb_8_hybrid',          'patch': None},
    'k=3 T=0.5':       {'dir': 'model/stock_emb_8_listmle_k3_t0.5', 'patch': None},
    'RankGLU':         {'dir': 'model/stock_emb_8_rankglu',         'patch': 'rankglu'},
    'EMA':             {'dir': 'model/stock_emb_8_ema',             'patch': None},
    'Multitask':       {'dir': 'model/stock_emb_8_multitask',       'patch': None},
    'GLU Rank':        {'dir': 'model/stock_emb_8_glurank',         'patch': 'glurank'},
    'EMA+ListMLE':     {'dir': 'model/stock_emb_8_ema_listmle',     'patch': None},
    'Contrastive':     {'dir': 'model/stock_emb_8_contrastive',     'patch': None},
    'SE v2':           {'dir': 'model/stock_emb_8_se_v2',           'patch': 'se_v2'},
}

# All 2-model ensembles with EMA and Hybrid as anchors
ENSEMBLES = [
    ['EMA', 'Hybrid dim=8'],
    ['EMA', 'Contrastive'],
    ['EMA', 'EMA+ListMLE'],
    ['EMA', 'GLU Rank'],
    ['Hybrid dim=8', 'Contrastive'],
    ['Hybrid dim=8', 'EMA+ListMLE'],
    ['Hybrid dim=8', 'GLU Rank'],
    ['EMA', 'SE v2'],
    ['Hybrid dim=8', 'SE v2'],
    ['EMA', 'RankGLU'],
    ['Hybrid dim=8', 'RankGLU'],
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


def load_experts_for_model(model_name, feature_dim, num_stocks, device):
    info = MODELS[model_name]
    model_dir = info['dir']
    patch_type = info['patch']

    import ensemble_models as _em
    _orig_trans_init = _em.StockTransformerExpert.__init__
    _orig_conv_init = _em.ConvStockExpert.__init__
    _orig_fa = _em.FeatureAttention

    if patch_type == 'rankglu':
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

    elif patch_type == 'glurank':
        class GatedLinear(nn.Module):
            def __init__(self, in_dim, out_dim, dropout=0.1):
                super().__init__()
                self.value = nn.Linear(in_dim, out_dim)
                self.gate = nn.Linear(in_dim, out_dim)
                self.dropout = nn.Dropout(dropout)
                self.norm = nn.LayerNorm(out_dim)
            def forward(self, x):
                v = self.value(x)
                g = torch.sigmoid(self.gate(x))
                return self.dropout(self.norm(v * g))
        def _patched_trans_init(self, input_dim, expert_config, num_stocks):
            _orig_trans_init(self, input_dim, expert_config, num_stocks)
            dm = self.d_model
            do = expert_config.get('dropout', 0.1)
            self.ranking_layers = nn.Sequential(
                GatedLinear(dm, dm, do), GatedLinear(dm, dm//2, do),
                nn.LayerNorm(dm//2), nn.ReLU(), nn.Dropout(do))
        def _patched_conv_init(self, input_dim, expert_config, num_stocks):
            _orig_conv_init(self, input_dim, expert_config, num_stocks)
            h = self.d_model
            do = expert_config.get('dropout', 0.1)
            self.ranking_layers = nn.Sequential(
                GatedLinear(h, h, do), GatedLinear(h, h//2, do),
                nn.LayerNorm(h//2), nn.ReLU(), nn.Dropout(do))
        _em.StockTransformerExpert.__init__ = _patched_trans_init
        _em.ConvStockExpert.__init__ = _patched_conv_init

    elif patch_type == 'se_v2':
        class SEBlock(nn.Module):
            def __init__(self, channels, reduction=4):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(channels, channels // reduction), nn.ReLU(),
                    nn.Linear(channels // reduction, channels), nn.Sigmoid())
            def forward(self, x):
                return x * self.fc(x)
        class SEWrapper(nn.Module):
            def __init__(self, fa_module, d_model):
                super().__init__()
                self.fa = fa_module
                self.se = SEBlock(d_model)
            def forward(self, x):
                return self.se(self.fa(x))
        def _se_trans_init(self, input_dim, expert_config, num_stocks):
            _orig_trans_init(self, input_dim, expert_config, num_stocks)
            self.feature_attention = SEWrapper(self.feature_attention, self.d_model)
        def _se_conv_init(self, input_dim, expert_config, num_stocks):
            _orig_conv_init(self, input_dim, expert_config, num_stocks)
            self.feature_attention = SEWrapper(self.feature_attention, self.d_model)
        _em.StockTransformerExpert.__init__ = _se_trans_init
        _em.ConvStockExpert.__init__ = _se_conv_init

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

    # Restore originals
    _em.FeatureAttention = _orig_fa
    _em.StockTransformerExpert.__init__ = _orig_trans_init
    _em.ConvStockExpert.__init__ = _orig_conv_init

    return models


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args_cpu, _ = parser.parse_known_args()
    device = torch.device('cpu') if args_cpu.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    set_seed(42)

    # Check which models are available
    available = {}
    for name in MODELS:
        info = MODELS[name]
        model_dir = info['dir']
        ok = os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, 'ensemble_config.json'))
        if ok:
            n_experts = len([f for f in os.listdir(model_dir) if f.startswith('expert_') and f.endswith('.pth')])
            ok = n_experts >= 4
        available[name] = ok
        print(f"  {name}: {'READY' if ok else 'MISSING'}")

    if not available.get('Hybrid dim=8'):
        print("ERROR: Hybrid baseline not available")
        return

    # Load data
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

    # Load all available models
    scorers = {}
    for name in MODELS:
        if not available[name]:
            continue
        print(f'\nLoading: {name}...')
        models = load_experts_for_model(name, feature_dim, num_stocks, device)
        print(f'  {len(models)} experts loaded')
        scorers[name] = models

    # Evaluate all 2-model ensembles
    pred_dates = ['2026-06-01', '2026-06-08']
    all_results = {}

    # First: standalone eval for each model
    print(f"\n{'='*60}")
    print("STANDALONE EVALUATION")
    for name in MODELS:
        if not available[name] or name in all_results:
            continue
        models = scorers.get(name)
        if not models:
            continue
        returns_list = []
        for pred_date_str in pred_dates:
            wr = _eval_models(models, pred_date_str, processed, features, raw_data, test_dates, test_df, device)
            if wr is not None:
                returns_list.append(wr)
                print(f'  [{name}] {pred_date_str}: {wr*100:+.2f}%')
        if returns_list:
            avg = np.mean(returns_list)
            all_results[name] = {'w1': returns_list[0]*100, 'w2': returns_list[-1]*100, 'avg': avg*100}

    # Then: ensemble evaluation
    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION")
    for ensemble_names in ENSEMBLES:
        label = ' + '.join(ensemble_names)
        if not all(available.get(n) for n in ensemble_names):
            continue

        returns_list = []
        for pred_date_str in pred_dates:
            all_model_scores = []
            for mname in ensemble_names:
                models = scorers[mname]
                mc_scores_list = []
                for _ in range(MC_PASSES):
                    pass_scores = []
                    for model in models:
                        with torch.no_grad():
                            seq_t, valid_mask, stock_ids = _build_sequence(
                                pred_date_str, processed, features, device)
                            if seq_t is None:
                                continue
                            pred = model(seq_t)
                            if isinstance(pred, tuple):
                                pred = pred[0]
                            pred = pred[0].cpu().numpy()
                        pass_scores.append(pred)
                    if pass_scores:
                        mc_scores_list.append(np.mean(pass_scores, axis=0))
                if mc_scores_list:
                    all_model_scores.append(np.mean(mc_scores_list, axis=0))

            if not all_model_scores or len(all_model_scores) < len(ensemble_names):
                continue
            ensemble_scores = np.mean(all_model_scores, axis=0)

            wr = _apply_postprocess(ensemble_scores, valid_mask, stock_ids, pred_date_str,
                                    raw_data, test_dates, test_df)
            if wr is not None:
                returns_list.append(wr)
                print(f'  [{label}] {pred_date_str}: {wr*100:+.2f}%')

        if len(returns_list) >= 2:
            avg = np.mean(returns_list)
            all_results[label] = {'w1': returns_list[0]*100, 'w2': returns_list[1]*100, 'avg': avg*100}
            print(f'  => W1: {returns_list[0]*100:+.2f}%  W2: {returns_list[1]*100:+.2f}%  Avg: {avg*100:+.2f}%')

    # Final ranking
    print(f'\n{"="*70}')
    print(f'{"Model/Ensemble":<40} {"Jun W1":>10} {"Jun W2":>10} {"Avg":>10}')
    print('-' * 70)
    ranked = sorted(all_results.items(), key=lambda x: x[1]['avg'], reverse=True)
    for name, r in ranked:
        print(f'{name:<40} {r["w1"]:>+9.2f}% {r["w2"]:>+9.2f}% {r["avg"]:>+9.2f}%')
    print('Done!')


def _build_sequence(pred_date_str, processed, features, device):
    pred_dt = pd.to_datetime(pred_date_str)
    hist = processed[processed['日期'] <= pred_date_str]
    avail_stocks = hist['股票代码'].unique()
    stock_ids = sorted(avail_stocks)
    n_stocks = len(stock_ids)
    n_feats = len(features)
    if n_stocks < 5:
        return None, None, None
    sequences = np.zeros((1, n_stocks, SEQUENCE_LENGTH, n_feats), dtype=np.float32)
    valid_mask = np.zeros(n_stocks, dtype=bool)
    for i, sid in enumerate(stock_ids):
        stock_data = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(stock_data) >= SEQUENCE_LENGTH:
            sequences[0, i] = stock_data[features].values[-SEQUENCE_LENGTH:].astype(np.float32)
            valid_mask[i] = True
    return torch.FloatTensor(sequences).to(device), valid_mask, stock_ids


def _apply_postprocess(scores, valid_mask, stock_ids, pred_date_str, raw_data, test_dates, test_df):
    raw_scores = {}
    for i, sid in enumerate(stock_ids):
        raw_scores[sid] = float(scores[i]) if valid_mask[i] else -float('inf')

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
    t5_range = [d for d in test_dates if d >= pd.to_datetime(pred_date_str)]
    t5_date = t5_range[min(4, len(t5_range) - 1)]

    rets = []
    for sid, weight in picks:
        t1 = test_df[(test_df['股票代码'] == sid) & (test_df['日期'] == t1_date)]
        t5 = test_df[(test_df['股票代码'] == sid) & (test_df['日期'] == t5_date)]
        if len(t1) == 0 or len(t5) == 0:
            r = 0.0
        else:
            r = (float(t5.iloc[0]['开盘']) - float(t1.iloc[0]['开盘'])) / float(t1.iloc[0]['开盘'])
        rets.append(r * weight)
    return sum(rets)


def _eval_models(models, pred_date_str, processed, features, raw_data, test_dates, test_df, device):
    seq_t, valid_mask, stock_ids = _build_sequence(pred_date_str, processed, features, device)
    if seq_t is None:
        return None
    all_scores = []
    for _ in range(MC_PASSES):
        pass_scores = []
        for model in models:
            with torch.no_grad():
                pred = model(seq_t)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pass_scores.append(pred[0].cpu().numpy())
        all_scores.append(np.mean(pass_scores, axis=0))
    mc_scores = np.mean(all_scores, axis=0)
    return _apply_postprocess(mc_scores, valid_mask, stock_ids, pred_date_str, raw_data, test_dates, test_df)


if __name__ == '__main__':
    main()
