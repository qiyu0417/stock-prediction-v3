"""Predict & verify Week of June 15-19, 2026 — sequential download (baostock is NOT thread-safe)"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib, time
import baostock as bs

print('='*60)
print('Week of June 15-19, 2026 — Prediction & Verification')
print('='*60)

# --- Step 1: Sequential download ---
print('\n[1/3] Downloading latest data...')
with open('data/stock_mapping.json') as f: mapping = json.load(f)
reverse = {v: k for k, v in mapping.items()}
real_codes = list(reverse.keys())[:300]
print(f'  {len(real_codes)} stocks')

bs.login()
all_new = []
for i, rc in enumerate(real_codes):
    bsc = f'sh.{rc}' if rc.startswith('6') else f'sz.{rc}'
    try:
        rs = bs.query_history_k_data_plus(bsc,
            'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg',
            start_date='2026-06-12', end_date='2026-06-20',
            frequency='d', adjustflag='1')
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        if rows:
            df_new = pd.DataFrame(rows, columns=rs.fields)
            for c in ['open','high','low','close','preclose','volume','amount','turn','pctChg']:
                df_new[c] = pd.to_numeric(df_new[c], errors='coerce')
            df_new['振幅'] = ((df_new['high']-df_new['low'])/df_new['preclose']*100).round(2)
            df_new['涨跌额'] = (df_new['close']-df_new['preclose']).round(2)
            df_new['code'] = rc
            df_new['date'] = pd.to_datetime(df_new['date'])
            df_new = df_new.rename(columns={'code':'股票代码','date':'日期','open':'开盘','close':'收盘',
                'high':'最高','low':'最低','volume':'成交量','amount':'成交额','turn':'换手率','pctChg':'涨跌幅'})
            all_new.append(df_new[['股票代码','日期','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌额','换手率','涨跌幅']])
    except Exception as e:
        pass
    if (i+1) % 50 == 0:
        print(f'    {i+1}/{len(real_codes)}...')

bs.logout()

new_data = pd.concat(all_new, ignore_index=True)
new_data['股票代码'] = new_data['股票代码'].map(reverse)
new_data = new_data.dropna(subset=['股票代码'])
new_dates = sorted(new_data['日期'].unique())
print(f'  Got {len(new_data)} rows, dates: {new_dates[0].date()} ~ {new_dates[-1].date()}')

# --- Step 2: Prepare data + load models ---
print('\n[2/3] Feature engineering + loading models...')
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda'); set_seed(42)
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6); train_df['日期'] = pd.to_datetime(train_df['日期'])
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6); test_df['日期'] = pd.to_datetime(test_df['日期'])

full_df = pd.concat([train_df, test_df, new_data], ignore_index=True)
full_df = full_df.drop_duplicates(subset=['股票代码', '日期'], keep='last')
full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
all_sids = sorted(full_df['股票代码'].unique()); sid2idx = {s:i for i,s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]; fcols_all = feature_cloums_map[FEATURE_NUM]

groups = [g.reset_index(drop=True) for _, g in full_df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument'] = processed_raw['股票代码'].map(sid2idx)
processed_raw = processed_raw.dropna(subset=['instrument']).copy()
processed_raw['instrument'] = processed_raw['instrument'].astype(np.int64)
processed_raw = _build_label_and_clean(processed_raw, drop_small_open=True)
alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]

import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__; _orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

def load_m(mdir, nf):
    _em.StockTransformerExpert.__init__ = _orig_ti; _em.ConvStockExpert.__init__ = _orig_ci
    with open(os.path.join(mdir, 'ensemble_config.json')) as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    with open('model/stock_emb_8_hybrid/ensemble_config.json') as f2: ns = json.load(f2)['num_stocks']
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(nf, e, ns) if ec['type']=='transformer' else ConvStockExpert(nf, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)
    return models

def preproc(mdir):
    with open(os.path.join(mdir, 'winsor_bounds.json')) as f: wb = json.load(f)
    sc = joblib.load(os.path.join(mdir, 'scaler.pkl'))
    p = processed_raw.copy()
    p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
    for col, (lo, hi) in wb.items():
        if col in p.columns: p[col] = p[col].clip(lo, hi)
    p[fcols_all] = sc.transform(p[fcols_all])
    return p

print('  Hybrid...'); H = load_m('model/stock_emb_8_hybrid', 197); p_h = preproc('model/stock_emb_8_hybrid')
print('  Alpha158...'); A = load_m('model/stock_emb_8_alpha158', len(alpha_f)); p_a = preproc('model/stock_emb_8_alpha158')

# --- Step 3: Predict ---
print('\n[3/3] Predicting & verifying...')
prediction_dates = sorted([d for d in processed_raw['日期'].unique() if d >= pd.to_datetime('2026-06-12')])

for pd_str in prediction_dates:
    pd_s = str(pd_str.date())
    recent = processed_raw[(processed_raw['日期'] <= pd_str) & (processed_raw['日期'] >= pd_str - pd.Timedelta(days=10))]
    daily_ret = recent.groupby('日期')['涨跌幅'].mean()
    up_pct = (daily_ret > 0).mean(); trend = daily_ret.mean() * 100

    use_alpha = trend > 0
    label = 'Alpha158' if use_alpha else 'Hybrid'
    p_now = p_a if use_alpha else p_h
    fcols_now = alpha_f if use_alpha else fcols_all
    M = A if use_alpha else H
    nf_now = len(alpha_f) if use_alpha else 197

    hist = p_now[p_now['日期'] <= pd_str]; sids = sorted(hist['股票代码'].unique())
    seq = np.zeros((1, len(sids), SEQ, nf_now), dtype=np.float32)
    varr = np.zeros(len(sids), dtype=bool)
    for i, sid in enumerate(sids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ: seq[0, i] = sd[fcols_now].values[-SEQ:].astype(np.float32); varr[i] = True

    seq_t = torch.FloatTensor(seq).to(device)
    all_s = []
    for _ in range(MC):
        ps = []
        for m in M:
            with torch.no_grad():
                out = m(seq_t)
                if isinstance(out, tuple): out = out[0]
                ps.append(out[0].cpu().numpy())
        all_s.append(np.mean(ps, axis=0))
    sc = np.mean(all_s, axis=0)

    raw_hist = processed_raw[processed_raw['日期'] <= pd_str]; sl = list(sids)
    filt = volatility_filter(raw_hist, sl, str(pd_str.date()), top_pct=VP)
    bnc = bounce_confirm(raw_hist, filt, str(pd_str.date()), threshold=BT)
    qual = compute_quality_score(raw_hist, filt, str(pd_str.date()))
    final = {}
    for i, sid in enumerate(sids):
        if not varr[i] or sid not in filt: continue
        s = float(sc[i])
        if sid not in bnc: s *= BP
        s += (qual.get(sid, 0.5) - 0.5) * QC
        final[sid] = s
    top5 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f'\n{pd_s} | {label} | up_pct={up_pct:.1%} trend={trend:+.2f}%')
    print(f'  Top-5: {[s for s,_ in top5]}')

    future_dates = sorted([d for d in new_dates if d > pd_str])
    if len(future_dates) >= 5:
        t1, t5 = future_dates[0], future_dates[min(4, len(future_dates)-1)]
        _, weights = equal_weight_allocate([s for s,_ in top5])
        actual_ret = 0.0
        for sid, w in zip([s for s,_ in top5], weights):
            r1 = new_data[(new_data['股票代码']==sid)&(new_data['日期']==t1)]
            r5 = new_data[(new_data['股票代码']==sid)&(new_data['日期']==t5)]
            if len(r1)>0 and len(r5)>0:
                sr = (float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])
                actual_ret += sr * w
                print(f'    {sid}: T+1={float(r1.iloc[0]["开盘"]):.2f} T+5={float(r5.iloc[0]["开盘"]):.2f} => {sr*100:+.2f}%')
            else:
                print(f'    {sid}: NO DATA')
        print(f'  >>> RETURN: {actual_ret*100:+.2f}%')
    else:
        print(f'  Only {len(future_dates)} future days')

print('\nDone!')
