"""#1 Conditional switching + #2 Asymmetric weighting — Alpha158 vs Hybrid"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20; SEQ = 60; device = torch.device('cuda'); set_seed(42)
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

# --- data ---
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
with open('model/stock_emb_8_hybrid/ensemble_config.json', 'r') as f: cfg = json.load(f)
fd_full = cfg['feature_dim']; ns = cfg['num_stocks']

def preprocess_models(mdir, n_feat):
    with open(os.path.join(mdir, 'winsor_bounds.json'), 'r') as f: wb = json.load(f)
    scaler = joblib.load(os.path.join(mdir, 'scaler.pkl'))
    p = processed_raw.copy()
    p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
    for col, (lo, hi) in wb.items():
        if col in p.columns: p[col] = p[col].clip(lo, hi)
    p[fcols_all] = scaler.transform(p[fcols_all])
    return p

import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__; _orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

def load_experts(mdir, n_feat):
    _em.StockTransformerExpert.__init__ = _orig_ti; _em.ConvStockExpert.__init__ = _orig_ci
    with open(os.path.join(mdir, 'ensemble_config.json'), 'r') as f: cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8); models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e = dict(ec); e['stock_embed_dim'] = emb
        m = StockTransformerExpert(n_feat, e, ns) if ec['type']=='transformer' else ConvStockExpert(n_feat, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.to(device); m.train()
        models.append(m)
    return models

def mc_infer(models, seq_t):
    all_s = []
    for _ in range(MC):
        ps = []
        for m in models:
            with torch.no_grad():
                out = m(seq_t)
                if isinstance(out, tuple): out = out[0]
                ps.append(out[0].cpu().numpy())
        all_s.append(np.mean(ps, axis=0))
    return np.mean(all_s, axis=0)

def compute_return(pd_str, sc, valid, sids):
    raw_hist = processed_raw[processed_raw['日期'] <= pd_str]
    sids_list = list(sids)
    filt = volatility_filter(raw_hist, sids_list, pd_str, top_pct=VP)
    bnc = bounce_confirm(raw_hist, filt, pd_str, threshold=BT)
    qual = compute_quality_score(raw_hist, filt, pd_str)
    final = {}
    for i, sid in enumerate(sids):
        if not valid[i] or sid not in filt: continue
        s = float(sc[i])
        if sid not in bnc: s *= BP
        s += (qual.get(sid, 0.5) - 0.5) * QC
        final[sid] = s
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
    return ret, [s for s,_ in picks]

# --- load models ---
print('Loading models...')
H = load_experts('model/stock_emb_8_hybrid', fd_full)
A = load_experts('model/stock_emb_8_alpha158', len(alpha_f))

# --- preprocess ---
p_h = preprocess_models('model/stock_emb_8_hybrid', fd_full)
p_a = preprocess_models('model/stock_emb_8_alpha158', len(alpha_f))

# --- compute MC=20 scores ---
print('Computing MC=20 scores...')
scores = {}
for pd_str in ['2026-06-01', '2026-06-08']:
    # Hybrid
    hist_h = p_h[p_h['日期'] <= pd_str]; sids_h = sorted(hist_h['股票代码'].unique())
    seq_h = np.zeros((1, len(sids_h), SEQ, fd_full), dtype=np.float32)
    v_h = np.zeros(len(sids_h), dtype=bool)
    for i, sid in enumerate(sids_h):
        sd = hist_h[hist_h['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ: seq_h[0, i] = sd[fcols_all].values[-SEQ:].astype(np.float32); v_h[i] = True
    sc_h = mc_infer(H, torch.FloatTensor(seq_h).to(device))

    # Alpha158
    hist_a = p_a[p_a['日期'] <= pd_str]; sids_a = sorted(hist_a['股票代码'].unique())
    seq_a = np.zeros((1, len(sids_a), SEQ, len(alpha_f)), dtype=np.float32)
    v_a = np.zeros(len(sids_a), dtype=bool)
    for i, sid in enumerate(sids_a):
        sd = hist_a[hist_a['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ: seq_a[0, i] = sd[alpha_f].values[-SEQ:].astype(np.float32); v_a[i] = True
    sc_a = mc_infer(A, torch.FloatTensor(seq_a).to(device))

    scores[pd_str] = {'sc_h': sc_h, 'v_h': v_h, 'sids_h': sids_h,
                       'sc_a': sc_a, 'v_a': v_a, 'sids_a': sids_a}

# --- market indicators at prediction time ---
def compute_market_indicators(pd_str):
    data = processed_raw[processed_raw['日期'] <= pd_str]
    recent = data[data['日期'] >= pd.to_datetime(pd_str) - pd.Timedelta(days=10)]
    daily_ret = recent.groupby('日期')['涨跌幅'].mean()
    up_pct = (daily_ret > 0).mean()
    trend = daily_ret.mean() * 100
    vol = daily_ret.std() * 100
    return up_pct, trend, vol

# --- baseline ---
print('\n=== Baselines ===')
for pd_str in ['2026-06-01', '2026-06-08']:
    s = scores[pd_str]
    wr_h, _ = compute_return(pd_str, s['sc_h'], s['v_h'], s['sids_h'])
    wr_a, _ = compute_return(pd_str, s['sc_a'], s['v_a'], s['sids_a'])
    up_pct, trend, vol = compute_market_indicators(pd_str)
    print(f'{pd_str}: Hybrid={wr_h*100:+.2f}%  Alpha={wr_a*100:+.2f}%  up_pct={up_pct:.1%}  trend={trend:+.2f}%  vol={vol:.2f}%')

# --- switching rules ---
print('\n=== #1: Conditional Switching Rules ===')
print(f'{"Rule":<40} {"W1":>8} {"W2":>8} {"Avg":>8}')

def eval_switch(pd_str, model_choice):
    s = scores[pd_str]
    if model_choice == 'H': return compute_return(pd_str, s['sc_h'], s['v_h'], s['sids_h'])
    elif model_choice == 'A': return compute_return(pd_str, s['sc_a'], s['v_a'], s['sids_a'])
    elif model_choice == 'avg':
        sc_avg = (s['sc_h'] + s['sc_a']) / 2.0
        return compute_return(pd_str, sc_avg, s['v_h'], s['sids_h'])

switching_rules = [
    ('Always Hybrid',           lambda u,t,v: 'H'),
    ('Always Alpha158',         lambda u,t,v: 'A'),
    ('Simple average A+H',      lambda u,t,v: 'avg'),
    ('If trend>0 → A else H',   lambda u,t,v: 'A' if t > 0 else 'H'),
    ('If up_pct>0.5 → A else H', lambda u,t,v: 'A' if u > 0.5 else 'H'),
    ('If up_pct>0.6 → A, <0.3 → H else avg', lambda u,t,v: 'A' if u > 0.6 else ('H' if u < 0.3 else 'avg')),
    ('If vol<1.5 → A else H',   lambda u,t,v: 'A' if v < 1.5 else 'H'),
    ('If trend>0 & up>0.5 → A else H', lambda u,t,v: 'A' if (t > 0 and u > 0.5) else 'H'),
    ('Oracle: A for W1, H for W2', None),  # handled separately
]

for rule_name, rule_fn in switching_rules:
    if rule_name.startswith('Oracle'):
        wr1, _ = eval_switch('2026-06-01', 'A')
        wr2, _ = eval_switch('2026-06-08', 'H')
    else:
        wr1, _ = eval_switch('2026-06-01', rule_fn(*compute_market_indicators('2026-06-01')))
        wr2, _ = eval_switch('2026-06-08', rule_fn(*compute_market_indicators('2026-06-08')))
    avg = (wr1 + wr2) / 2
    print(f'{rule_name:<40} {wr1*100:+7.2f}% {wr2*100:+7.2f}% {avg*100:+7.2f}%')

# --- asymmetric weights ---
print('\n=== #2: Asymmetric Weighting ===')
print(f'{"Scheme":<45} {"W1":>8} {"W2":>8} {"Avg":>8}')

weight_schemes = [
    ('Simple avg (w=0.5)',                 lambda u,t,v: 0.5),
    ('w_A = clip(up_pct, 0.3, 0.7)',       lambda u,t,v: np.clip(u, 0.3, 0.7)),
    ('w_A = 0.5 + trend*0.02',             lambda u,t,v: np.clip(0.5 + t*0.02, 0.2, 0.8)),
    ('w_A = 0.4 + up_pct*0.3',             lambda u,t,v: 0.4 + u*0.3),
    ('w_A = 0.6 if trend>0 else 0.3',      lambda u,t,v: 0.6 if t > 0 else 0.3),
    ('w_A = 0.7 if up_pct>0.5 else 0.3',   lambda u,t,v: 0.7 if u > 0.5 else 0.3),
    ('w_A = 0.3 (Hybrid dominant)',         lambda u,t,v: 0.3),
    ('w_A = 0.7 (Alpha dominant)',          lambda u,t,v: 0.7),
    ('w_A = 0.4 if vol>1.5 else 0.6',      lambda u,t,v: 0.4 if v > 1.5 else 0.6),
    ('Oracle: w_A=0.8(W1) 0.2(W2)',        None),
]

for scheme_name, w_fn in weight_schemes:
    if scheme_name.startswith('Oracle'):
        w_a1, w_a2 = 0.8, 0.2
    else:
        up1, t1, v1 = compute_market_indicators('2026-06-01')
        up2, t2, v2 = compute_market_indicators('2026-06-08')
        w_a1, w_a2 = w_fn(up1, t1, v1), w_fn(up2, t2, v2)

    for pd_str, w_a in [('2026-06-01', w_a1), ('2026-06-08', w_a2)]:
        s = scores[pd_str]
        sc_w = w_a * s['sc_a'] + (1 - w_a) * s['sc_h']
        if pd_str == '2026-06-01':
            wr1, _ = compute_return(pd_str, sc_w, s['v_h'], s['sids_h'])
        else:
            wr2, _ = compute_return(pd_str, sc_w, s['v_h'], s['sids_h'])
    avg = (wr1 + wr2) / 2
    print(f'{scheme_name:<45} {wr1*100:+7.2f}% {wr2*100:+7.2f}% {avg*100:+7.2f}%')

print('\nDone!')
