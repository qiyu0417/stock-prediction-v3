"""Compare equal-weight vs asymmetric weighting schemes across 3 weeks"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score

MC=20; SEQ=60; device=torch.device('cuda'); set_seed(42)
VP,BT,BP,QC=0.95,0.008,0.92,0.05

print('Loading data...')
train_df=pd.read_csv('data/train.csv',dtype={'股票代码':str});train_df['股票代码']=train_df['股票代码'].astype(str).str.zfill(6);train_df['日期']=pd.to_datetime(train_df['日期'],format='mixed')
test_df=pd.read_csv('data/test.csv',dtype={'股票代码':str});test_df['股票代码']=test_df['股票代码'].astype(str).str.zfill(6);test_df['日期']=pd.to_datetime(test_df['日期'],format='mixed')
new_df=pd.read_csv('data/new_week.csv',dtype={'股票代码':str});new_df['股票代码']=new_df['股票代码'].astype(str).str.zfill(6);new_df['日期']=pd.to_datetime(new_df['日期'],format='mixed')
full_df=pd.concat([train_df,test_df,new_df]).drop_duplicates(subset=['股票代码','日期'],keep='last')
full_df=full_df.sort_values(['股票代码','日期']).reset_index(drop=True)
all_sids=sorted(full_df['股票代码'].unique());sid2idx={s:i for i,s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe=feature_engineer_func_map[FEATURE_NUM];fcols_all=feature_cloums_map[FEATURE_NUM]
groups=[g.reset_index(drop=True) for _,g in full_df.groupby('股票代码',sort=False) if len(g)>=SEQ+10]
processed_raw=pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument']=processed_raw['股票代码'].map(sid2idx)
processed_raw=processed_raw.dropna(subset=['instrument']).copy();processed_raw['instrument']=processed_raw['instrument'].astype(np.int64)
processed_raw=_build_label_and_clean(processed_raw,drop_small_open=True)
alpha_f=[f for f in _ALPHA_158_COLS if f in fcols_all]

with open('model/stock_emb_8_hybrid/ensemble_config.json','r') as f: ns=json.load(f)['num_stocks']

import ensemble_models as _em
_orig_ti=_em.StockTransformerExpert.__init__;_orig_ci=_em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert,ConvStockExpert

def load_m(mdir,nf):
    _em.StockTransformerExpert.__init__=_orig_ti;_em.ConvStockExpert.__init__=_orig_ci
    with open(os.path.join(mdir,'ensemble_config.json')) as f: cfg=json.load(f)
    emb=cfg.get('stock_embed_dim',8);models=[]
    for ec in cfg['expert_configs']:
        path=os.path.join(mdir,f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e=dict(ec);e['stock_embed_dim']=emb
        m=StockTransformerExpert(nf,e,ns) if ec['type']=='transformer' else ConvStockExpert(nf,e,ns)
        m.load_state_dict(torch.load(path,map_location=device,weights_only=True));m.to(device);m.train()
        models.append(m)
    return models

def preproc(mdir):
    with open(os.path.join(mdir,'winsor_bounds.json')) as f: wb=json.load(f)
    sc=joblib.load(os.path.join(mdir,'scaler.pkl'))
    p=processed_raw.copy();p[fcols_all]=p[fcols_all].replace([np.inf,-np.inf],np.nan).dropna(subset=fcols_all)
    for col,(lo,hi) in wb.items():
        if col in p.columns: p[col]=p[col].clip(lo,hi)
    p[fcols_all]=sc.transform(p[fcols_all])
    return p

def mc_infer(models,seq_t):
    all_s=[]
    for _ in range(MC):
        ps=[]
        for m in models:
            with torch.no_grad():
                out=m(seq_t)
                if isinstance(out,tuple): out=out[0]
                ps.append(out[0].cpu().numpy())
        all_s.append(np.mean(ps,axis=0))
    return np.mean(all_s,axis=0)

def get_top5(sc,valid,sids,pd_str):
    raw_hist=processed_raw[processed_raw['日期']<=pd_str];sl=list(sids)
    filt=volatility_filter(raw_hist,sl,str(pd_str.date()),top_pct=VP)
    bnc=bounce_confirm(raw_hist,filt,str(pd_str.date()),threshold=BT)
    qual=compute_quality_score(raw_hist,filt,str(pd_str.date()))
    final={}
    for i,sid in enumerate(sids):
        if not valid[i] or sid not in filt: continue
        s=float(sc[i])
        if sid not in bnc: s*=BP
        s+=(qual.get(sid,0.5)-0.5)*QC;final[sid]=s
    return sorted(final.items(),key=lambda x:x[1],reverse=True)[:5]

def compute_ret(stocks,weights,pd_str,t1,t5):
    ret=0.0;all_data=pd.concat([test_df,new_df])
    for sid,w in zip(stocks,weights):
        r1=all_data[(all_data['股票代码']==sid)&(all_data['日期']==t1)]
        r5=all_data[(all_data['股票代码']==sid)&(all_data['日期']==t5)]
        if len(r1)>0 and len(r5)>0:
            ret+=(float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])*w
    return ret

# --- schemes that depend on scores ---
def make_weights(scores, scheme):
    """scores = numpy array of 5 scores (higher=better)"""
    n=5
    if scheme=='equal':        return np.full(n,0.2)
    if scheme=='score_prop':   return scores/scores.sum() if scores.sum()>0 else np.full(n,0.2)
    if scheme=='softmax':
        s=np.clip(scores,-10,10);s=s-np.max(s);e=np.exp(s);return e/e.sum()
    # Rank-based (rank 1=best)
    ranks=np.argsort(np.argsort(-scores))+1
    if scheme=='inv_rank':     w=1.0/ranks;return w/w.sum()
    if scheme=='inv_sqrt':     w=1.0/np.sqrt(ranks);return w/w.sum()
    if scheme=='linear_30':    return np.array([0.30,0.25,0.20,0.15,0.10])
    if scheme=='top40':        return np.array([0.40,0.25,0.15,0.12,0.08])
    if scheme=='top50':        return np.array([0.50,0.20,0.15,0.10,0.05])
    if scheme=='moderate':     return np.array([0.25,0.22,0.20,0.18,0.15])
    return np.full(n,0.2)

# --- load models ---
print('Loading models...')
H=load_m('model/stock_emb_8_hybrid',197);p_h=preproc('model/stock_emb_8_hybrid')
A=load_m('model/stock_emb_8_alpha158',len(alpha_f));p_a=preproc('model/stock_emb_8_alpha158')

weeks=[
    ('W1',pd.to_datetime('2026-05-29'),pd.to_datetime('2026-06-01'),pd.to_datetime('2026-06-05')),
    ('W2',pd.to_datetime('2026-06-05'),pd.to_datetime('2026-06-08'),pd.to_datetime('2026-06-12')),
    ('W3',pd.to_datetime('2026-06-12'),pd.to_datetime('2026-06-15'),pd.to_datetime('2026-06-18')),
]

# --- get all top-5 picks and scores ---
print('Computing MC=20 scores...')
all_weeks={}
for wname,pd_str,t1,t5 in weeks:
    pd_s=str(pd_str.date())
    # Hybrid
    hist=p_h[p_h['日期']<=pd_s];sids=sorted(hist['股票代码'].unique())
    seq=np.zeros((1,len(sids),SEQ,197),dtype=np.float32);v=np.zeros(len(sids),dtype=bool)
    for i,sid in enumerate(sids):
        sd=hist[hist['股票代码']==sid].sort_values('日期')
        if len(sd)>=SEQ: seq[0,i]=sd[fcols_all].values[-SEQ:].astype(np.float32);v[i]=True
    sc_h=mc_infer(H,torch.FloatTensor(seq).to(device))
    top5_h=get_top5(sc_h,v,sids,pd_str)
    scores_h=np.array([s for _,s in top5_h])

    # Alpha158
    hist=p_a[p_a['日期']<=pd_s];sids=sorted(hist['股票代码'].unique())
    seq=np.zeros((1,len(sids),SEQ,len(alpha_f)),dtype=np.float32);v=np.zeros(len(sids),dtype=bool)
    for i,sid in enumerate(sids):
        sd=hist[hist['股票代码']==sid].sort_values('日期')
        if len(sd)>=SEQ: seq[0,i]=sd[alpha_f].values[-SEQ:].astype(np.float32);v[i]=True
    sc_a=mc_infer(A,torch.FloatTensor(seq).to(device))
    top5_a=get_top5(sc_a,v,sids,pd_str)
    scores_a=np.array([s for _,s in top5_a])

    all_weeks[wname]={'top5_h':top5_h,'scores_h':scores_h,'top5_a':top5_a,'scores_a':scores_a,'t1':t1,'t5':t5}

# ====== EVALUATION ======
schemes=['equal','score_prop','softmax','inv_rank','inv_sqrt','linear_30','top40','top50','moderate']
scheme_names={'equal':'Equal 20-20-20-20-20','score_prop':'Score Proportional','softmax':'Softmax Scores',
    'inv_rank':'1/rank','inv_sqrt':'1/sqrt(rank)',
    'linear_30':'TopHeavy 30-25-20-15-10','top40':'Winner 40-25-15-12-8',
    'top50':'Extreme 50-20-15-10-5','moderate':'Moderate 25-22-20-18-15'}

print('\n'+'='*75)
print(f'{"Scheme":<30} {"Hybrid":>14} {"Alpha158":>14}')
print('='*75)

best_h=(0,'');best_a=(0,'')
for scheme in schemes:
    rets_h=[];rets_a=[]
    for wname,data in all_weeks.items():
        wh=make_weights(data['scores_h'],scheme)
        rets_h.append(compute_ret([s for s,_ in data['top5_h']],wh,None,data['t1'],data['t5']))
        wa=make_weights(data['scores_a'],scheme)
        rets_a.append(compute_ret([s for s,_ in data['top5_a']],wa,None,data['t1'],data['t5']))
    avg_h=np.mean(rets_h);avg_a=np.mean(rets_a)
    if avg_h>best_h[0]: best_h=(avg_h,scheme)
    if avg_a>best_a[0]: best_a=(avg_a,scheme)
    h_mark=' < BEST' if avg_h==best_h[0] else ''
    a_mark=' < BEST' if avg_a==best_a[0] else ''
    print(f'{scheme_names[scheme]:<30} {avg_h*100:+6.2f}%{h_mark:>8} {avg_a*100:+6.2f}%{a_mark:>8}')

# --- detail for best schemes ---
print('\n'+'='*75)
print('DETAIL: Weekly breakdown for best schemes')
print('='*75)
for model_name, best_scheme, weeks_key in [
    ('Hybrid', best_h[1], 'top5_h'),
    ('Alpha158', best_a[1], 'top5_a'),
]:
    print(f'\n{model_name} + {scheme_names[best_scheme]}:')
    for wname,data in all_weeks.items():
        top5=data[weeks_key]
        ws=make_weights(data[f'scores_{weeks_key[-1]}'],best_scheme)
        ret=compute_ret([s for s,_ in top5],ws,None,data['t1'],data['t5'])
        stocks_str=' | '.join([f'{s}({w*100:.0f}%)' for (s,_),w in zip(top5,ws)])
        print(f'  {wname}: {ret*100:+.2f}%  [{stocks_str}]')

# --- also show W1/W2/W3 per-scheme detail ---
print('\n'+'='*75)
print('PER-WEEK: All schemes for both models')
print('='*75)
print(f'{"Scheme":<25} {"W1":>20} {"W2":>20} {"W3":>20} {"Avg":>20}')
for model_label, key, scores_key in [('--HYBRID--','top5_h','scores_h'),('--ALPHA158--','top5_a','scores_a')]:
    print(f'{model_label}')
    for scheme in schemes:
        rets=[]
        for wname,data in all_weeks.items():
            ws=make_weights(data[scores_key],scheme)
            rets.append(compute_ret([s for s,_ in data[key]],ws,None,data['t1'],data['t5']))
        avg=np.mean(rets)
        print(f'  {scheme_names[scheme]:<23} {rets[0]*100:+7.2f}% (W1) {rets[1]*100:+7.2f}% (W2) {rets[2]*100:+7.2f}% (W3) {avg*100:+7.2f}%')

print('\nDone!')
