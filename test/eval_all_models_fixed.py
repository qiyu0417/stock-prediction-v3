"""Re-evaluate ALL models with FIXED post-processing (processed data) + 5-seed MC=20"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS, _TECH_39_ONLY
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda')
SEEDS = [42, 123, 456, 789, 1024]

# Post-processing params (OldDefault — verified optimal)
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

# --- data (one-time) ---
print('Loading data...')
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str}); train_df['股票代码']=train_df['股票代码'].astype(str).str.zfill(6); train_df['日期']=pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str}); test_df['股票代码']=test_df['股票代码'].astype(str).str.zfill(6); test_df['日期']=pd.to_datetime(test_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=['股票代码', '日期'], keep='last')
test_dates = sorted(test_df['日期'].unique())
all_sids = sorted(full_df['股票代码'].unique()); sid2idx = {s:i for i,s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]; fcols_all = feature_cloums_map[FEATURE_NUM]

df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument'] = processed_raw['股票代码'].map(sid2idx); processed_raw = processed_raw.dropna(subset=['instrument']).copy()
processed_raw['instrument'] = processed_raw['instrument'].astype(np.int64)
processed_raw = _build_label_and_clean(processed_raw, drop_small_open=True)

# Feature subsets
alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]

# Load Hybrid config for ns and stock_embed_dim
with open('model/stock_emb_8_hybrid/ensemble_config.json', 'r') as f: cfg = json.load(f)
ns = cfg['num_stocks']

# --- model loading ---
import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__; _orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

def load_model(mdir, fd):
    _em.StockTransformerExpert.__init__ = _orig_ti; _em.ConvStockExpert.__init__ = _orig_ci
    with open(os.path.join(mdir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8); models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(fd, e, ns) if ec['type']=='transformer' else ConvStockExpert(fd, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)
    return models

# Model registry
MODELS = {
    'Hybrid':        ('model/stock_emb_8_hybrid',     197),
    'EMA':           ('model/stock_emb_8_ema',         197),
    'EMA+ListMLE':   ('model/stock_emb_8_ema_listmle', 197),
    'SE_v2':         ('model/stock_emb_8_se_v2',       197),
    'Alpha158':      ('model/stock_emb_8_alpha158',    169),
}

# --- eval ---
def eval_one_seed(model_name, model_dir, n_feat, fcols, seed, post_process=True):
    set_seed(seed)
    models = load_model(model_dir, n_feat)

    # Preprocess with model-specific winsor+scaler
    with open(os.path.join(model_dir, 'winsor_bounds.json'), 'r') as f: wb = json.load(f)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    p = processed_raw.copy()
    p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
    for col, (lo, hi) in wb.items():
        if col in p.columns: p[col] = p[col].clip(lo, hi)
    p[fcols_all] = scaler.transform(p[fcols_all])

    results = {}
    for pd_str in ['2026-06-01', '2026-06-08']:
        hist = p[p['日期'] <= pd_str]
        sids = sorted(hist['股票代码'].unique()); n_stocks = len(sids)

        seq = np.zeros((1, n_stocks, SEQ, n_feat), dtype=np.float32)
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
            # Post-process with processed_raw data (correct!)
            raw_hist = processed_raw[processed_raw['日期'] <= pd_str]
            sids_list = list(sids)

            if VP < 1.0:
                filt = volatility_filter(raw_hist, sids_list, pd_str, top_pct=VP)
            else:
                filt = sids_list
            bnc = bounce_confirm(raw_hist, filt, pd_str, threshold=BT)
            qual = compute_quality_score(raw_hist, filt, pd_str) if QC > 0 else {}

            final = {}
            for i, sid in enumerate(sids):
                if not valid[i] or sid not in filt: continue
                s = float(sc[i])
                if sid not in bnc: s *= BP
                s += (qual.get(sid, 0.5) - 0.5) * QC
                final[sid] = s
        else:
            final = {sid: float(sc[i]) for i, sid in enumerate(sids) if valid[i]}

        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
        picks = list(zip(*equal_weight_allocate([s for s,_ in ranked])))

        t1 = next(d for d in test_dates if d >= pd.to_datetime(pd_str))
        t5r = [d for d in test_dates if d >= pd.to_datetime(pd_str)]; t5 = t5r[min(4, len(t5r)-1)]
        ret = 0.0
        for sid, w in picks:
            r1 = test_df[(test_df['股票代码']==sid)&(test_df['日期']==t1)]
            r5 = test_df[(test_df['股票代码']==sid)&(test_df['日期']==t5)]
            if len(r1)>0 and len(r5)>0:
                ret += (float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])*w
        results[pd_str] = ret
    del models; torch.cuda.empty_cache()
    return (results['2026-06-01'] + results['2026-06-08']) / 2, results['2026-06-01'], results['2026-06-08']

print(f'\nEvaluating {len(MODELS)} models x 5 seeds x 2 modes (with/without post-processing)...')
print(f'Total: {len(MODELS)*5*2} MC=20 runs (~{len(MODELS)*5*2*3}min)\n')

all_results = {}

for model_name, (mdir, n_feat) in MODELS.items():
    if model_name == 'Alpha158':
        fcols = alpha_f
    else:
        fcols = fcols_all

    print(f'--- {model_name} ({n_feat}dim) ---')
    with_pp = []
    without_pp = []

    for seed in SEEDS:
        avg_w, w1, w2 = eval_one_seed(model_name, mdir, n_feat, fcols, seed, post_process=True)
        with_pp.append((avg_w, w1, w2))
        avg_n, w1n, w2n = eval_one_seed(model_name, mdir, n_feat, fcols, seed, post_process=False)
        without_pp.append((avg_n, w1n, w2n))
        print(f'  seed={seed}: w/PP={avg_w*100:+.2f}%  w/o PP={avg_n*100:+.2f}%')

    w_avgs = [r[0] for r in with_pp]
    wo_avgs = [r[0] for r in without_pp]
    w_mean = np.mean(w_avgs); w_std = np.std(w_avgs)
    wo_mean = np.mean(wo_avgs); wo_std = np.std(wo_avgs)

    print(f'  w/PP:  mean={w_mean*100:+.2f}%  std={w_std*100:.2f}%  range=[{min(w_avgs)*100:+.2f}~{max(w_avgs)*100:+.2f}]%')
    print(f'  w/o PP: mean={wo_mean*100:+.2f}%  std={wo_std*100:.2f}%  range=[{min(wo_avgs)*100:+.2f}~{max(wo_avgs)*100:+.2f}]%')
    print(f'  Δ: {w_mean*100 - wo_mean*100:+.2f}%')

    all_results[model_name] = {
        'with_pp_mean': w_mean, 'with_pp_std': w_std,
        'without_pp_mean': wo_mean, 'without_pp_std': wo_std,
        'with_pp_best': max(w_avgs), 'with_pp_worst': min(w_avgs),
    }

print('\n' + '='*80)
print('FINAL RANKING (5-seed mean, with post-processing)')
print('='*80)
print(f'{"Model":<18} {"w/PP Avg":>10} {"Std":>8} {"Range":>20} {"w/o Avg":>10} {"Δ":>8}')
print('-'*80)
ranked = sorted(all_results.items(), key=lambda x: x[1]['with_pp_mean'], reverse=True)
for name, r in ranked:
    print(f'{name:<18} {r["with_pp_mean"]*100:+8.2f}% {r["with_pp_std"]*100:>7.2f}% '
          f'[{r["with_pp_worst"]*100:+.2f}~{max(r["with_pp_mean"]+r["with_pp_std"], r["with_pp_best"])*100:+.2f}]% '
          f'{r["without_pp_mean"]*100:+8.2f}% {r["with_pp_mean"]*100 - r["without_pp_mean"]*100:+7.2f}%')

print('\nDone!')
