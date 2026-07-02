"""Continue eval for SE_v2 and Alpha158 with fixed post-processing"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib, torch.nn as nn
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda')
SEEDS = [42, 123, 456, 789, 1024]
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

# Capture ORIGINAL __init__ ONCE at module level
import ensemble_models as _em
from ensemble_models import StockTransformerExpert, ConvStockExpert
ORIG_TI = StockTransformerExpert.__init__
ORIG_CI = ConvStockExpert.__init__

from train_se_v2 import SEBlock, SEWrapper

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

alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]

with open('model/stock_emb_8_hybrid/ensemble_config.json', 'r') as f: ns = json.load(f)['num_stocks']

def eval_model(model_name, model_dir, n_feat, fcols, seeds, use_se_patch=False):
    with_pp = []; without_pp = []
    with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    with open(os.path.join(model_dir, 'winsor_bounds.json'), 'r') as f: wb = json.load(f)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

    for seed in seeds:
        set_seed(seed); torch.cuda.empty_cache()
        # ALWAYS reset to original before each seed
        _em.StockTransformerExpert.__init__ = ORIG_TI
        _em.ConvStockExpert.__init__ = ORIG_CI

        if use_se_patch:
            def se_init_ti(self_m, fd, ec, ns_stk):
                ORIG_TI(self_m, fd, ec, ns_stk)
                self_m.feature_attention = SEWrapper(self_m.feature_attention, self_m.d_model)
            def se_init_ci(self_m, fd, ec, ns_stk):
                ORIG_CI(self_m, fd, ec, ns_stk)
                self_m.feature_attention = SEWrapper(self_m.feature_attention, self_m.d_model)
            _em.StockTransformerExpert.__init__ = se_init_ti
            _em.ConvStockExpert.__init__ = se_init_ci

        models = []
        for ec in cfg['expert_configs']:
            path = os.path.join(model_dir, f'expert_{ec["name"]}.pth')
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

        for do_pp in [True, False]:
            avg_ret = 0.0
            for pd_str in ['2026-06-01', '2026-06-08']:
                hist = p[p['日期'] <= pd_str]
                sids = sorted(hist['股票代码'].unique()); n_stocks = len(sids)
                seq = np.zeros((1, n_stocks, SEQ, n_feat), dtype=np.float32)
                varr = np.zeros(n_stocks, dtype=bool)
                for i, sid in enumerate(sids):
                    sd = hist[hist['股票代码'] == sid].sort_values('日期')
                    if len(sd) >= SEQ:
                        seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32)
                        varr[i] = True
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

                if do_pp:
                    raw_hist = processed_raw[processed_raw['日期'] <= pd_str]
                    sids_list = list(sids)
                    if VP < 1.0: filt = volatility_filter(raw_hist, sids_list, pd_str, top_pct=VP)
                    else: filt = sids_list
                    bnc = bounce_confirm(raw_hist, filt, pd_str, threshold=BT)
                    qual = compute_quality_score(raw_hist, filt, pd_str) if QC > 0 else {}
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
                avg_ret += ret / 2.0
            if do_pp: with_pp.append(avg_ret)
            else: without_pp.append(avg_ret)

        print(f'  seed={seed}: w/PP={with_pp[-1]*100:+.2f}%  w/o PP={without_pp[-1]*100:+.2f}%')
        del models; torch.cuda.empty_cache()

    w_mean = np.mean(with_pp); w_std = np.std(with_pp)
    wo_mean = np.mean(without_pp); wo_std = np.std(without_pp)
    print(f'  w/PP:  mean={w_mean*100:+.2f}%  std={w_std*100:.2f}%')
    print(f'  w/o PP: mean={wo_mean*100:+.2f}%  std={wo_std*100:.2f}%')
    print(f'  Delta: {w_mean*100 - wo_mean*100:+.2f}%')
    return w_mean, w_std, wo_mean, wo_std

print('\n--- SE_v2 (197dim, with SE patch) ---')
se_w, se_ws, se_wo, se_wos = eval_model('SE_v2', 'model/stock_emb_8_se_v2', 197, fcols_all, SEEDS, use_se_patch=True)

print('\n--- Alpha158 (169dim) ---')
a_w, a_ws, a_wo, a_wos = eval_model('Alpha158', 'model/stock_emb_8_alpha158', 169, alpha_f, SEEDS, use_se_patch=False)

print('\n' + '='*80)
print('FINAL RANKING (5 models x 5 seeds, with post-processing)')
print('='*80)
print(f'{"Model":<18} {"w/PP Avg":>10} {"Std":>8} {"w/o Avg":>10} {"Delta":>8}')
print('-'*60)
results = [
    ('Hybrid',        8.59, 0.49, 8.01, 0.72),
    ('EMA+ListMLE',   8.33, 0.51, 8.02, 0.59),
    ('EMA',           8.07, 0.57, 8.33, 1.40),
    ('SE_v2',        se_w*100, se_ws*100, se_wo*100, se_wos*100),
    ('Alpha158',     a_w*100, a_ws*100, a_wo*100, a_wos*100),
]
for name, wm, ws, wom, wos in sorted(results, key=lambda x: x[1], reverse=True):
    delta = wm - wom
    print(f'{name:<18} {wm:+8.2f}% ±{ws:.2f}%  {wom:+8.2f}% ±{wos:.2f}%  Δ{delta:+7.2f}%')

print('\nDone!')
