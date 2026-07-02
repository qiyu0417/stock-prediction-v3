"""Compare T5-open label vs old label (close_t5) — MC=20, fixed PP, seed=42"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda'); set_seed(42)
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

print('Loading data...')
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

with open('model/stock_emb_8_t5open/ensemble_config.json', 'r') as f: ns = json.load(f)['num_stocks']

import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__; _orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

def load_and_eval(label, mdir, n_feat, fcols):
    _em.StockTransformerExpert.__init__ = _orig_ti; _em.ConvStockExpert.__init__ = _orig_ci
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

    results = {}
    for pd_str in ['2026-06-01', '2026-06-08']:
        hist = p[p['日期'] <= pd_str]; sids = sorted(hist['股票代码'].unique())
        seq = np.zeros((1, len(sids), SEQ, n_feat), dtype=np.float32)
        varr = np.zeros(len(sids), dtype=bool)
        for i, sid in enumerate(sids):
            sd = hist[hist['股票代码'] == sid].sort_values('日期')
            if len(sd) >= SEQ: seq[0,i] = sd[fcols].values[-SEQ:].astype(np.float32); varr[i] = True

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

        raw_hist = processed_raw[processed_raw['日期'] <= pd_str]; sids_list = list(sids)
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
        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
        picks = list(zip(*equal_weight_allocate([s for s,_ in ranked])))
        t1 = next(d for d in test_dates if d >= pd.to_datetime(pd_str))
        t5r = [d for d in test_dates if d >= pd.to_datetime(pd_str)]; t5 = t5r[min(4, len(t5r)-1)]
        ret = sum((float(test_df[(test_df['股票代码']==sid)&(test_df['日期']==t5)].iloc[0]['开盘'])-float(test_df[(test_df['股票代码']==sid)&(test_df['日期']==t1)].iloc[0]['开盘']))/float(test_df[(test_df['股票代码']==sid)&(test_df['日期']==t1)].iloc[0]['开盘'])*w for sid, w in picks if len(test_df[(test_df['股票代码']==sid)&(test_df['日期']==t1)])>0 and len(test_df[(test_df['股票代码']==sid)&(test_df['日期']==t5)])>0)
        results[pd_str] = (ret, [s for s,_ in picks])
    return results

print('\n--- OLD label (close_t5) ---')
old = load_and_eval('OLD', 'model/stock_emb_8_hybrid', 197, fcols_all)
print(f'  W1: {old["2026-06-01"][1][:3]} | {old["2026-06-01"][0]*100:+.2f}%')
print(f'  W2: {old["2026-06-08"][1][:3]} | {old["2026-06-08"][0]*100:+.2f}%')
avg_old = (old['2026-06-01'][0] + old['2026-06-08'][0]) / 2

print('\n--- NEW label (open_t5) ---')
new = load_and_eval('NEW', 'model/stock_emb_8_t5open', 197, fcols_all)
print(f'  W1: {new["2026-06-01"][1][:3]} | {new["2026-06-01"][0]*100:+.2f}%')
print(f'  W2: {new["2026-06-08"][1][:3]} | {new["2026-06-08"][0]*100:+.2f}%')
avg_new = (new['2026-06-01'][0] + new['2026-06-08'][0]) / 2

print(f'\n  OLD avg: {avg_old*100:+.2f}%')
print(f'  NEW avg: {avg_new*100:+.2f}%')
print(f'  DELTA:   {(avg_new-avg_old)*100:+.2f}%')
print('\nDone!')
