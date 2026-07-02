"""Evaluate SWA model: 5-seed MC=20 across 3 June weeks + comparison with Hybrid/Alpha158"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda')
SEEDS = [42, 123, 456, 789, 1024]
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

# --- data ---
print('Loading data...')
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str}); train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6); train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str}); test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6); test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')
new_df = pd.read_csv('data/new_week.csv', dtype={'股票代码': str}); new_df['股票代码'] = new_df['股票代码'].astype(str).str.zfill(6); new_df['日期'] = pd.to_datetime(new_df['日期'], format='mixed')
all_data = pd.concat([test_df, new_df])
full_df = pd.concat([train_df, test_df, new_df]).drop_duplicates(subset=['股票代码', '日期'], keep='last')
full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
all_sids = sorted(full_df['股票代码'].unique()); sid2idx = {s: i for i, s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]; fcols_all = feature_cloums_map[FEATURE_NUM]
groups = [g.reset_index(drop=True) for _, g in full_df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument'] = processed_raw['股票代码'].map(sid2idx)
processed_raw = processed_raw.dropna(subset=['instrument']).copy(); processed_raw['instrument'] = processed_raw['instrument'].astype(np.int64)
processed_raw = _build_label_and_clean(processed_raw, drop_small_open=True)

import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__; _orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

with open('model/stock_emb_8_hybrid/ensemble_config.json') as f: ns = json.load(f)['num_stocks']

def load_model(mdir, nf):
    _em.StockTransformerExpert.__init__ = _orig_ti; _em.ConvStockExpert.__init__ = _orig_ci
    with open(os.path.join(mdir, 'ensemble_config.json')) as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(nf, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(nf, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)
    return models

def preproc(mdir):
    with open(os.path.join(mdir, 'winsor_bounds.json')) as f: wb = json.load(f)
    sc = joblib.load(os.path.join(mdir, 'scaler.pkl'))
    p = processed_raw.copy(); p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
    for col, (lo, hi) in wb.items():
        if col in p.columns: p[col] = p[col].clip(lo, hi)
    p[fcols_all] = sc.transform(p[fcols_all])
    return p

# 3 weeks
weeks = [
    ('W1', pd.to_datetime('2026-05-29'), pd.to_datetime('2026-06-01'), pd.to_datetime('2026-06-05'), 5),
    ('W2', pd.to_datetime('2026-06-05'), pd.to_datetime('2026-06-08'), pd.to_datetime('2026-06-12'), 5),
    ('W3', pd.to_datetime('2026-06-12'), pd.to_datetime('2026-06-15'), pd.to_datetime('2026-06-18'), 4),
]

def eval_one(model_name, mdir, nf, fcols, seed, post_process=True):
    set_seed(seed)
    models = load_model(mdir, nf)
    p = preproc(mdir)

    results = {}
    for wname, pd_str, t1, t5, ndays in weeks:
        hist = p[p['日期'] <= pd_str]; sids = sorted(hist['股票代码'].unique())
        n_stocks = len(sids)
        seq = np.zeros((1, n_stocks, SEQ, nf), dtype=np.float32)
        valid = np.zeros(n_stocks, dtype=bool)
        for i, sid in enumerate(sids):
            sd = hist[hist['股票代码'] == sid].sort_values('日期')
            if len(sd) >= SEQ:
                seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32)
                valid[i] = True

        seq_t = torch.FloatTensor(seq).to(device)
        all_s = []
        for _ in range(MC):
            ps = []
            for m in models:
                with torch.no_grad():
                    out = m(seq_t)
                    if isinstance(out, tuple): out = out[0]
                    ps.append(out[0].cpu().numpy())
            all_s.append(np.mean(ps, axis=0))
        sc = np.mean(all_s, axis=0)

        if post_process:
            raw_hist = processed_raw[processed_raw['日期'] <= pd_str]; sl = list(sids)
            filt = volatility_filter(raw_hist, sl, str(pd_str.date()), top_pct=VP)
            bnc = bounce_confirm(raw_hist, filt, str(pd_str.date()), threshold=BT)
            qual = compute_quality_score(raw_hist, filt, str(pd_str.date()))
            final = {}
            for i, sid in enumerate(sids):
                if not valid[i] or sid not in filt: continue
                s = float(sc[i])
                if sid not in bnc: s *= BP
                s += (qual.get(sid, 0.5) - 0.5) * QC; final[sid] = s
        else:
            final = {sid: float(sc[i]) for i, sid in enumerate(sids) if valid[i]}

        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
        _, weights = equal_weight_allocate([s for s, _ in ranked])

        ret = 0.0
        for sid, w in zip([s for s, _ in ranked], weights):
            r1 = all_data[(all_data['股票代码'] == sid) & (all_data['日期'] == t1)]
            r5 = all_data[(all_data['股票代码'] == sid) & (all_data['日期'] == t5)]
            if len(r1) > 0 and len(r5) > 0:
                ret += (float(r5.iloc[0]['开盘']) - float(r1.iloc[0]['开盘'])) / float(r1.iloc[0]['开盘']) * w
        results[wname] = ret
    del models; torch.cuda.empty_cache()
    return results

# --- Evaluate SWA ---
print('\n' + '='*60)
print('SWA 5-seed MC=20 Evaluation')
print('='*60)

swa_dir = 'model/stock_emb_8_swa'
swa_pp = {s: {} for s in SEEDS}
swa_np = {s: {} for s in SEEDS}

for seed in SEEDS:
    r_pp = eval_one('SWA', swa_dir, 197, fcols_all, seed, post_process=True)
    r_np = eval_one('SWA', swa_dir, 197, fcols_all, seed, post_process=False)
    for wk in ['W1', 'W2', 'W3']:
        swa_pp[seed][wk] = r_pp.get(wk, 0)
        swa_np[seed][wk] = r_np.get(wk, 0)
    avg_pp = np.mean([swa_pp[seed][w] for w in ['W1', 'W2', 'W3']])
    avg_np = np.mean([swa_np[seed][w] for w in ['W1', 'W2', 'W3']])
    print(f'  seed={seed}: w/PP={avg_pp*100:+.2f}%  w/o PP={avg_np*100:+.2f}%')

# --- Summary ---
print('\n' + '='*80)
print('SWA WEEKLY BREAKDOWN (5-seed mean)')
print('='*80)
for wk in ['W1', 'W2', 'W3']:
    pp_vals = [swa_pp[s][wk] for s in SEEDS]
    np_vals = [swa_np[s][wk] for s in SEEDS]
    print(f'{wk}: w/PP={np.mean(pp_vals)*100:+.2f}%+-{np.std(pp_vals)*100:.2f}%  w/o PP={np.mean(np_vals)*100:+.2f}%+-{np.std(np_vals)*100:.2f}%')

pp_avgs = [np.mean([swa_pp[s][w] for w in ['W1', 'W2', 'W3']]) for s in SEEDS]
np_avgs = [np.mean([swa_np[s][w] for w in ['W1', 'W2', 'W3']]) for s in SEEDS]
print(f'\nOverall: w/PP={np.mean(pp_avgs)*100:+.2f}%+-{np.std(pp_avgs)*100:.2f}%  w/o PP={np.mean(np_avgs)*100:+.2f}%+-{np.std(np_avgs)*100:.2f}%')

# --- Compare with Hybrid ---
print('\n' + '='*80)
print('COMPARISON: SWA vs Hybrid (both 5-seed MC=20)')
print('='*80)

hybrid_dir = 'model/stock_emb_8_hybrid'
hybrid_pp = {s: {} for s in SEEDS}
for seed in SEEDS:
    r = eval_one('Hybrid', hybrid_dir, 197, fcols_all, seed, post_process=True)
    for wk in ['W1', 'W2', 'W3']:
        hybrid_pp[seed][wk] = r.get(wk, 0)
    avg = np.mean([hybrid_pp[seed][w] for w in ['W1', 'W2', 'W3']])
    print(f'  Hybrid seed={seed}: avg={avg*100:+.2f}%')

hybrid_pp_avgs = [np.mean([hybrid_pp[s][w] for w in ['W1', 'W2', 'W3']]) for s in SEEDS]
print(f'\nSWA  w/PP:  {np.mean(pp_avgs)*100:+.2f}% +- {np.std(pp_avgs)*100:.2f}%')
print(f'Hybrid w/PP: {np.mean(hybrid_pp_avgs)*100:+.2f}% +- {np.std(hybrid_pp_avgs)*100:.2f}%')
print(f'Delta: {(np.mean(pp_avgs) - np.mean(hybrid_pp_avgs))*100:+.2f}%')

print('\nDone!')
