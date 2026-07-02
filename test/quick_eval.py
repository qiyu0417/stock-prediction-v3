"""Quick eval: Weighted vs ListMLE"""
import sys, os, json
sys.path.insert(0, 'code/src')
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm

from config_v5 import SEQUENCE_LENGTH, MAX_STOCKS_PER_CHUNK, USE_AMP, MC_SAMPLES
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, _build_label_and_clean
from market_regime import compute_market_regime
from quality_filter import bounce_confirm, compute_quality_score, volatility_filter, equal_weight_allocate

ROOT = '.'
TEST_CUTOFFS = {
    'Jun W1': {'cutoff': '2026-05-29', 't1_open': '2026-06-01', 't5_open': '2026-06-05'},
    'Jun W2': {'cutoff': '2026-06-05', 't1_open': '2026-06-08', 't5_open': '2026-06-12'},
}

device = torch.device('cuda')
print(f'Device: {device}')

# Load raw data once
train = pd.read_csv('data/train.csv', dtype={'股票代码': str})
train['股票代码'] = train['股票代码'].str.zfill(6)
train['日期'] = pd.to_datetime(train['日期'].str.replace(' 00:00:00', ''), format='mixed')
test = pd.read_csv('data/test.csv', dtype={'股票代码': str})
test['股票代码'] = test['股票代码'].str.zfill(6)
test['日期'] = pd.to_datetime(test['日期'].str.replace(' 00:00:00', ''), format='mixed')
all_data = pd.concat([train, test], ignore_index=True).sort_values(['股票代码', '日期']).reset_index(drop=True)

def prepare_eval_data(df, winsor, scaler):
    """与 train.py 的预处理对齐, 包含 instrument 列"""
    fc = feature_cloums_map['158+39']  # includes 'instrument'
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    p = pd.concat([engineer_features_158plus39(g) for g in tqdm(groups, desc='FE')]).reset_index(drop=True)
    p['日期'] = pd.to_datetime(p['日期'].astype(str).str.replace(' 00:00:00', ''), format='mixed')

    stock_ids = sorted(p['股票代码'].unique())
    sid2idx = {s: i for i, s in enumerate(stock_ids)}
    p['instrument'] = p['股票代码'].map(sid2idx)

    # Winsorize
    for col, (lo, hi) in winsor.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)

    # Ensure features exist
    for col in fc:
        if col not in p.columns:
            p[col] = 0

    # Transform via numpy to avoid sklearn feature name check
    X = p[fc].values.astype(np.float64)
    p[fc] = scaler.transform(X)
    return p, fc


model_dirs = {
    'Weighted (baseline)': 'model/stock_emb_8_ensemble',
    'ListMLE': 'model/stock_emb_8_listmle',
}

results = {}
for name, model_dir in model_dirs.items():
    cfg_path = f'{model_dir}/ensemble_config.json'
    if not os.path.exists(cfg_path):
        print(f'{name}: SKIP (no config)')
        continue

    with open(cfg_path) as f:
        cfg = json.load(f)
    winsor = json.load(open(f'{model_dir}/winsor_bounds.json'))
    scaler = joblib.load(f'{model_dir}/scaler.pkl')

    # Load models
    models = []
    for ec in cfg['expert_configs']:
        path = f'{model_dir}/expert_{ec["name"]}.pth'
        if not os.path.exists(path):
            continue
        if ec['type'] == 'transformer':
            m = StockTransformerExpert(cfg.get('feature_dim', 197), ec, cfg.get('num_stocks', 300))
        else:
            m = ConvStockExpert(cfg.get('feature_dim', 197), ec, cfg.get('num_stocks', 300))
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        models.append(m)
    print(f'{name}: {len(models)} experts')

    pdata, features = prepare_eval_data(all_data.copy(), winsor, scaler)
    results[name] = {'models': models, 'pdata': pdata, 'features': features}

# Evaluate each week
for label_w, info in TEST_CUTOFFS.items():
    cutoff_dt = pd.to_datetime(info['cutoff'])
    print(f'\n--- {label_w} (cutoff={info["cutoff"]}) ---')

    for name, res in results.items():
        stock_ids = sorted(res['pdata']['股票代码'].unique())

        # Build sequences
        seqs, sids = [], []
        for sid in stock_ids:
            hist = res['pdata'][(res['pdata']['股票代码'] == sid) & (res['pdata']['日期'] <= cutoff_dt)]
            hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
            if len(hist) == SEQUENCE_LENGTH:
                seqs.append(hist[res['features']].values.astype(np.float32))
                sids.append(sid)

        if not seqs:
            print(f'  {name}: no sequences')
            continue

        x = torch.FloatTensor(np.stack(seqs)).unsqueeze(0).to(device)

        # MC predict
        models = res['models']
        n_models = len(models)
        all_rounds = []
        for seed in [42, 142, 242, 342, 442]:
            torch.manual_seed(seed); np.random.seed(seed)
            rnd = []
            for model in models:
                model.train()
                mc = []
                with torch.no_grad():
                    for _ in range(MC_SAMPLES):
                        with torch.amp.autocast('cuda', enabled=USE_AMP):
                            if x.size(1) <= MAX_STOCKS_PER_CHUNK:
                                s = model(x).squeeze(0)
                            else:
                                s = torch.cat([model(x[:, i:i+MAX_STOCKS_PER_CHUNK]).squeeze(0)
                                             for i in range(0, x.size(1), MAX_STOCKS_PER_CHUNK)], dim=0)
                        mc.append(s.cpu().numpy())
                rnd.append(np.mean(mc, axis=0))
            fused = np.zeros_like(rnd[0])
            for s in rnd:
                fused += s / n_models
            all_rounds.append(fused)
        raw = np.mean(all_rounds, axis=0)
        raw_scores = {sid: float(raw[i]) for i, sid in enumerate(sids)}

        # Post-processing
        regime = compute_market_regime(all_data, res['features'], sids, cutoff_dt)
        if regime.get('skip_trading', False):
            print(f'  {name}: SKIP (market regime)')
            continue

        kept = volatility_filter(all_data, sids, cutoff_dt, top_pct=0.95)
        confirmed = bounce_confirm(all_data, kept, cutoff_dt, threshold=0.008)
        quality = compute_quality_score(all_data, kept, cutoff_dt)

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
                sd = all_data[(all_data['股票代码'] == sid) & (all_data['日期'].dt.strftime('%Y-%m-%d') == info['t1_open'])]
                if len(sd) > 0:
                    t1_map[sid] = float(sd['开盘'].iloc[0])
                sd5 = all_data[(all_data['股票代码'] == sid) & (all_data['日期'].dt.strftime('%Y-%m-%d') == info['t5_open'])]
                if len(sd5) > 0:
                    t5_map[sid] = float(sd5['开盘'].iloc[0])
            stock_rets = []
            for sid in selected:
                if sid in t1_map and sid in t5_map and t1_map[sid] > 0:
                    stock_rets.append((t5_map[sid] - t1_map[sid]) / t1_map[sid])
            if stock_rets:
                ret = np.mean(stock_rets)

        print(f'  {name}: Top5={selected} | return={ret:+.4f} ({ret*100:+.2f}%)')

print('\n=== DONE ===')
