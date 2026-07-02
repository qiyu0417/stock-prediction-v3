"""5-seed x 3 models W1/W2 breakdown (seed=42/456/789, MC=20, fixed PP)"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib, torch.nn as nn
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda')
SEEDS = [42, 456, 789]

train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str}); train_df['股票代码']=train_df['股票代码'].astype(str).str.zfill(6); train_df['日期']=pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str}); test_df['股票代码']=test_df['股票代码'].astype(str).str.zfill(6); test_df['日期']=pd.to_datetime(test_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=['股票代码', '日期'], keep='last')
test_dates = sorted(test_df['日期'].unique()); all_sids = sorted(full_df['股票代码'].unique()); sid2idx = {s:i for i,s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]; fcols_all = feature_cloums_map[FEATURE_NUM]
df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument'] = processed_raw['股票代码'].map(sid2idx); processed_raw = processed_raw.dropna(subset=['instrument']).copy()
processed_raw['instrument'] = processed_raw['instrument'].astype(np.int64)
processed_raw = _build_label_and_clean(processed_raw, drop_small_open=True)

alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]
with open('model/stock_emb_8_hybrid/ensemble_config.json', 'r') as f: ns = json.load(f)['num_stocks']

import ensemble_models as _em
from ensemble_models import StockTransformerExpert, ConvStockExpert
ORIG_TI = StockTransformerExpert.__init__; ORIG_CI = ConvStockExpert.__init__
from train_se_v2 import SEWrapper

VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

def eval_seed_weekly(name, mdir, n_feat, fcols, seed, use_se=False):
    set_seed(seed); torch.cuda.empty_cache()
    _em.StockTransformerExpert.__init__ = ORIG_TI; _em.ConvStockExpert.__init__ = ORIG_CI
    if use_se:
        def si_ti(s, fd, ec, stk): ORIG_TI(s, fd, ec, stk); s.feature_attention = SEWrapper(s.feature_attention, s.d_model)
        def si_ci(s, fd, ec, stk): ORIG_CI(s, fd, ec, stk); s.feature_attention = SEWrapper(s.feature_attention, s.d_model)
        _em.StockTransformerExpert.__init__ = si_ti; _em.ConvStockExpert.__init__ = si_ci

    with open(os.path.join(mdir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    with open(os.path.join(mdir, 'winsor_bounds.json'), 'r') as f: wb = json.load(f)
    scaler = joblib.load(os.path.join(mdir, 'scaler.pkl'))

    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(n_feat, e, ns) if ec['type']=='transformer' else ConvStockExpert(n_feat, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)

    p = processed_raw.copy()
    p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
    for col, (lo, hi) in wb.items():
        if col in p.columns: p[col] = p[col].clip(lo, hi)
    p[fcols_all] = scaler.transform(p[fcols_all])

    w1_raw, w1_pp, w2_raw, w2_pp = None, None, None, None
    for pd_str in ['2026-06-01', '2026-06-08']:
        hist = p[p['日期'] <= pd_str]
        sids = sorted(hist['股票代码'].unique()); n_stocks = len(sids)
        seq = np.zeros((1, n_stocks, SEQ, n_feat), dtype=np.float32)
        varr = np.zeros(n_stocks, dtype=bool)
        for i, sid in enumerate(sids):
            sd = hist[hist['股票代码'] == sid].sort_values('日期')
            if len(sd) >= SEQ: seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32); varr[i] = True

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

        for do_pp, label in [(False, 'raw'), (True, 'pp')]:
            if do_pp:
                raw_hist = processed_raw[processed_raw['日期'] <= pd_str]
                sids_list = list(sids)
                filt = volatility_filter(raw_hist, sids_list, pd_str, top_pct=VP)
                bnc = bounce_confirm(raw_hist, filt, pd_str, threshold=BT)
                qual = compute_quality_score(raw_hist, filt, pd_str)
                final = {}
                for i, sid in enumerate(sids):
                    if not varr[i] or sid not in filt: continue
                    s = float(sc[i])
                    if sid not in bnc: s *= BP
                    s += (qual.get(sid, 0.5) - 0.5) * QC
                    final[sid] = s
            else:
                final = {sid: float(sc[i]) for i, sid in enumerate(sids) if varr[i]}

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

            if pd_str == '2026-06-01':
                if do_pp: w1_pp = ret
                else: w1_raw = ret
            else:
                if do_pp: w2_pp = ret
                else: w2_raw = ret

    return w1_raw, w1_pp, w2_raw, w2_pp

models_config = [
    ('Hybrid',       'model/stock_emb_8_hybrid',      197, fcols_all, False),
    ('EMA',          'model/stock_emb_8_ema',          197, fcols_all, False),
    ('EMA+ListMLE',  'model/stock_emb_8_ema_listmle',  197, fcols_all, False),
    ('Alpha158',     'model/stock_emb_8_alpha158',     169, alpha_f,   False),
    ('SE_v2',        'model/stock_emb_8_se_v2',        197, fcols_all, True),
]

print(f'Seeds: {SEEDS}')
print(f'{"Model":<16} {"Seed":<6} {"W1_raw":>8} {"W1_PP":>8} {"W2_raw":>8} {"W2_PP":>8}')
print('-' * 64)

# Collect per-seed data
all_data = {name: {'w1_raw': [], 'w1_pp': [], 'w2_raw': [], 'w2_pp': []} for name,_,_,_,_ in models_config}

for name, mdir, nf, fc, use_se in models_config:
    for seed in SEEDS:
        w1r, w1p, w2r, w2p = eval_seed_weekly(name, mdir, nf, fc, seed, use_se)
        all_data[name]['w1_raw'].append(w1r); all_data[name]['w1_pp'].append(w1p)
        all_data[name]['w2_raw'].append(w2r); all_data[name]['w2_pp'].append(w2p)
        print(f'{name:<16} {seed:<6} {w1r*100:+7.2f}% {w1p*100:+7.2f}% {w2r*100:+7.2f}% {w2p*100:+7.2f}%')

print()
print('='*80)
print('SUMMARY (3 seeds x MC=20, with PP)')
print('='*80)
print(f'{"Model":<16} {"W1_mean":>8} {"W2_mean":>8} {"Avg":>8} {"W1_range":>18} {"W2_range":>18}')
print('-'*80)
for name, d in sorted(all_data.items(), key=lambda x: np.mean(x[1]['w1_pp']+np.array(x[1]['w2_pp']))/2, reverse=True):
    w1m = np.mean(d['w1_pp']); w2m = np.mean(d['w2_pp'])
    avg = (w1m + w2m) / 2
    w1rng = f'[{min(d["w1_pp"])*100:+.2f}~{max(d["w1_pp"])*100:+.2f}]%'
    w2rng = f'[{min(d["w2_pp"])*100:+.2f}~{max(d["w2_pp"])*100:+.2f}]%'
    print(f'{name:<16} {w1m*100:+7.2f}% {w2m*100:+7.2f}% {avg*100:+7.2f}% {w1rng:>18} {w2rng:>18}')

print('\nDone!')
