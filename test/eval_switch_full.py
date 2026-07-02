"""#53: Conditional switching + asymmetric weighting — 3 weeks + stock-level weights"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score

MC=20; SEQ=60; device=torch.device('cuda'); set_seed(42)
VP,BT,BP,QC=0.95,0.008,0.92,0.05

# --- data (include new_week for W3) ---
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

def get_top5_final(sc,valid,sids,pd_str):
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

def compute_ret(top5_weights,pd_str,t1_date,t5_date):
    """top5_weights = [(sid, weight), ...]"""
    ret=0.0
    all_data=pd.concat([test_df,new_df])
    for sid,w in top5_weights:
        r1=all_data[(all_data['股票代码']==sid)&(all_data['日期']==t1_date)]
        r5=all_data[(all_data['股票代码']==sid)&(all_data['日期']==t5_date)]
        if len(r1)>0 and len(r5)>0:
            ret+=(float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])*w
    return ret

# --- market indicators ---
def market_ind(pd_str):
    data=processed_raw[processed_raw['日期']<=pd_str]
    recent=data[data['日期']>=pd_str-pd.Timedelta(days=10)]
    dr=recent.groupby('日期')['涨跌幅'].mean()
    return (dr>0).mean(), dr.mean()*100, dr.std()*100

# --- load models ---
print('Loading models...')
H=load_m('model/stock_emb_8_hybrid',197);p_h=preproc('model/stock_emb_8_hybrid')
A=load_m('model/stock_emb_8_alpha158',len(alpha_f));p_a=preproc('model/stock_emb_8_alpha158')

# --- 3-week config ---
weeks=[
    ('W1',pd.to_datetime('2026-05-29'),pd.to_datetime('2026-06-01'),pd.to_datetime('2026-06-05')),
    ('W2',pd.to_datetime('2026-06-05'),pd.to_datetime('2026-06-08'),pd.to_datetime('2026-06-12')),
    ('W3',pd.to_datetime('2026-06-12'),pd.to_datetime('2026-06-15'),pd.to_datetime('2026-06-18')),
]

# --- compute all scores ---
print('Computing MC=20 scores...')
scores={}
for wname,pd_str,t1,t5 in weeks:
    pd_s=str(pd_str.date())
    # Hybrid
    hist_h=p_h[p_h['日期']<=pd_s];sids_h=sorted(hist_h['股票代码'].unique())
    seq_h=np.zeros((1,len(sids_h),SEQ,197),dtype=np.float32);vh=np.zeros(len(sids_h),dtype=bool)
    for i,sid in enumerate(sids_h):
        sd=hist_h[hist_h['股票代码']==sid].sort_values('日期')
        if len(sd)>=SEQ: seq_h[0,i]=sd[fcols_all].values[-SEQ:].astype(np.float32);vh[i]=True
    sc_h=mc_infer(H,torch.FloatTensor(seq_h).to(device))
    top5_h=get_top5_final(sc_h,vh,sids_h,pd_str)
    raw_h={};valid_h={}
    for i,sid in enumerate(sids_h):
        if vh[i]: raw_h[sid]=float(sc_h[i]);valid_h[sid]=True

    # Alpha158
    hist_a=p_a[p_a['日期']<=pd_s];sids_a=sorted(hist_a['股票代码'].unique())
    seq_a=np.zeros((1,len(sids_a),SEQ,len(alpha_f)),dtype=np.float32);va=np.zeros(len(sids_a),dtype=bool)
    for i,sid in enumerate(sids_a):
        sd=hist_a[hist_a['股票代码']==sid].sort_values('日期')
        if len(sd)>=SEQ: seq_a[0,i]=sd[alpha_f].values[-SEQ:].astype(np.float32);va[i]=True
    sc_a=mc_infer(A,torch.FloatTensor(seq_a).to(device))
    top5_a=get_top5_final(sc_a,va,sids_a,pd_str)
    raw_a={};valid_a={}
    for i,sid in enumerate(sids_a):
        if va[i]: raw_a[sid]=float(sc_a[i]);valid_a[sid]=True

    # Combined score (0.5/0.5 blend)
    sc_combined={}
    for sid in set(list(raw_h.keys())+list(raw_a.keys())):
        sh=raw_h.get(sid,np.nan);sa=raw_a.get(sid,np.nan)
        if np.isnan(sh): sc_combined[sid]=sa
        elif np.isnan(sa): sc_combined[sid]=sh
        else: sc_combined[sid]=0.5*sh+0.5*sa
    # Re-rank combined (simplified post-processing: just sort)
    top5_combined=sorted(sc_combined.items(),key=lambda x:x[1],reverse=True)[:5]

    scores[wname]={'top5_h':top5_h,'top5_a':top5_a,'top5_c':top5_combined,
                   'raw_h':raw_h,'raw_a':raw_a,'valid_h':valid_h,'valid_a':valid_a}
    # Market
    up,trend,vol=market_ind(pd_str)
    print(f'{wname}: up_pct={up:.1%} trend={trend:+.1f}% vol={vol:.1f}%')
    print(f'  Hybrid:  {[s for s,_ in top5_h]}')
    print(f'  Alpha:   {[s for s,_ in top5_a]}')
    print(f'  Combine: {[s for s,_ in top5_combined]}')

# ====== EVALUATION ======
print('\n'+'='*70)
print('#1 CONDITIONAL SWITCHING')
print('='*70)

# Each week: which model's top-5 to use
def eval_week(wname,top5):
    idx=['W1','W2','W3'].index(wname)
    _,_,t1,t5=weeks[idx]
    return compute_ret([(s,0.2) for s,_ in top5],None,t1,t5)

print(f'{"Rule":<42} {"W1":>8} {"W2":>8} {"W3":>8} {"Avg":>8}')
for label, rule_fn in [
    ('Always Hybrid',           lambda u,t,v: 'H'),
    ('Always Alpha158',         lambda u,t,v: 'A'),
    ('Always Combined(0.5)',    lambda u,t,v: 'C'),
    ('If trend>0 -> A else H',  lambda u,t,v: 'A' if t>0 else 'H'),
    ('If up_pct>0.5 -> A else H',lambda u,t,v: 'A' if u>0.5 else 'H'),
    ('If trend>0 & up>0.5 -> A else H',lambda u,t,v: 'A' if (t>0 and u>0.5) else 'H'),
    ('If vol<1.5 -> A else H',  lambda u,t,v: 'H' if v>1.5 else 'A'),
    ('Oracle (best each week)', None),
]:
    rets=[]
    for wname in ['W1','W2','W3']:
        up,trend,vol=market_ind(pd.to_datetime(
            {'W1':'2026-05-29','W2':'2026-06-05','W3':'2026-06-12'}[wname]))
        if label.startswith('Oracle'):
            choice={'W1':'H','W2':'H','W3':'A'}[wname]
        else:
            choice=rule_fn(up,trend,vol)
        top5_key={'H':'top5_h','A':'top5_a','C':'top5_c'}[choice]
        top5=scores[wname][top5_key]
        rets.append(eval_week(wname,top5))
    avg=np.mean(rets)
    print(f'{label:<42} {rets[0]*100:+7.2f}% {rets[1]*100:+7.2f}% {rets[2]*100:+7.2f}% {avg*100:+7.2f}%')

# ====== STOCK-LEVEL ASYMMETRIC WEIGHTING ======
print('\n'+'='*70)
print('#2 STOCK-LEVEL ASYMMETRIC WEIGHTING (on Hybrid top-5)')
print('='*70)

def weight_schemes(top5):
    """top5 = [(sid, score), ...] returns list of weight schemes"""
    scores=np.array([s for _,s in top5])
    n=5
    results={}
    # Equal
    results['Equal']=[1.0/n]*n
    # Score proportional
    if scores.sum()>0:
        results['ScoreProp']=list(scores/scores.sum())
    # Softmax
    s_exp=np.exp(np.clip(scores,-10,10)-np.max(np.clip(scores,-10,10)))
    results['Softmax']=list(s_exp/s_exp.sum())
    # Top-heavy: 30/25/20/15/10
    results['TopHeavy 30-10']=[0.30,0.25,0.20,0.15,0.10]
    # Super top: 40/20/17/13/10
    results['SuperTop 40-10']=[0.40,0.20,0.17,0.13,0.10]
    # Moderate: 25/22/20/18/15
    results['Moderate 25-15']=[0.25,0.22,0.20,0.18,0.15]
    # Rank-based: weight ~ 1/sqrt(rank)
    raw=1.0/np.sqrt(np.arange(1,n+1));results['InvSqrtRank']=list(raw/raw.sum())
    return results

print(f'{"Scheme":<25} {"W1":>8} {"W2":>8} {"W3":>8} {"Avg":>8}')
for model_name, top5_key in [('Hybrid','top5_h'),('Alpha158','top5_a')]:
    print(f'-- {model_name} --')
    for ws_name, ws_fn in [
        ('Equal',None),('ScoreProp',None),('Softmax',None),
        ('TopHeavy 30-10',None),('SuperTop 40-10',None),
        ('InvSqrtRank',None),
    ]:
        rets=[]
        for wname, (_,_,t1,t5) in zip(['W1','W2','W3'],weeks):
            top5=scores[wname][top5_key]
            weights=weight_schemes(top5)[ws_name] if ws_name!='Equal' else [0.2]*5
            pairs=list(zip([s for s,_ in top5],weights))
            rets.append(compute_ret(pairs,None,t1,t5))
        avg=np.mean(rets)
        marker=' <--' if avg>0.10 else ''
        print(f'  {ws_name:<23} {rets[0]*100:+7.2f}% {rets[1]*100:+7.2f}% {rets[2]*100:+7.2f}% {avg*100:+7.2f}%{marker}')

# ====== COMBINED: Switch model + Asymmetric stock weights ======
print('\n'+'='*70)
print('#3 COMBINED: Best switch rule + Best weight scheme')
print('='*70)

# Best switch: trend>0 -> Alpha else Hybrid
# Best weights: try TopHeavy or ScoreProp on selected model
for wname in ['W1','W2','W3']:
    up,trend,vol=market_ind(pd.to_datetime(
        {'W1':'2026-05-29','W2':'2026-06-05','W3':'2026-06-12'}[wname]))
    choice='A' if trend>0 else 'H'
    top5=scores[wname][f'top5_{choice.lower()[0]}']
    _,_,t1,t5=weeks[['W1','W2','W3'].index(wname)]

    rets_eq=compute_ret([(s,0.2) for s,_ in top5],None,t1,t5)
    weights=weight_schemes(top5)
    best_w=None;best_r=-99
    for wn,ws in weights.items():
        r=compute_ret(list(zip([s for s,_ in top5],ws)),None,t1,t5)
        if r>best_r: best_r=r;best_w=wn
    print(f'{wname}: switch->{choice}  Equal={rets_eq*100:+.2f}%  BestW={best_w}->{best_r*100:+.2f}%')

print('\nDone!')
