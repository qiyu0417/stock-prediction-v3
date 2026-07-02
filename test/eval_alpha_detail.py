"""Alpha158 vs Hybrid detail — stock picks, top-10 scores, per-week returns"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC=20; SEQ=60; device=torch.device('cuda'); set_seed(42)
VP,BT,BP,QC=0.95,0.008,0.92,0.05

print('Loading...')
train_df=pd.read_csv('data/train.csv',dtype={'股票代码':str});train_df['股票代码']=train_df['股票代码'].astype(str).str.zfill(6);train_df['日期']=pd.to_datetime(train_df['日期'],format='mixed')
test_df=pd.read_csv('data/test.csv',dtype={'股票代码':str});test_df['股票代码']=test_df['股票代码'].astype(str).str.zfill(6);test_df['日期']=pd.to_datetime(test_df['日期'],format='mixed')
full_df=pd.concat([train_df,test_df]).drop_duplicates(subset=['股票代码','日期'],keep='last')
test_dates=sorted(test_df['日期'].unique());all_sids=sorted(full_df['股票代码'].unique());sid2idx={s:i for i,s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe=feature_engineer_func_map[FEATURE_NUM];fcols_all=feature_cloums_map[FEATURE_NUM]
df=full_df.sort_values(['股票代码','日期']).reset_index(drop=True)
groups=[g.reset_index(drop=True) for _,g in df.groupby('股票代码',sort=False) if len(g)>=SEQ+10]
processed_raw=pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument']=processed_raw['股票代码'].map(sid2idx);processed_raw=processed_raw.dropna(subset=['instrument']).copy()
processed_raw['instrument']=processed_raw['instrument'].astype(np.int64)
processed_raw=_build_label_and_clean(processed_raw,drop_small_open=True)

alpha_f=[f for f in _ALPHA_158_COLS if f in fcols_all]
with open('model/stock_emb_8_hybrid/ensemble_config.json') as f: ns=json.load(f)['num_stocks']

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
    scaler=joblib.load(os.path.join(mdir,'scaler.pkl'))
    p=processed_raw.copy()
    p[fcols_all]=p[fcols_all].replace([np.inf,-np.inf],np.nan).dropna(subset=fcols_all)
    for col,(lo,hi) in wb.items():
        if col in p.columns: p[col]=p[col].clip(lo,hi)
    p[fcols_all]=scaler.transform(p[fcols_all])
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

def eval_model(p,pd_str,fcols,n_feat,models,label):
    hist=p[p['日期']<=pd_str];sids=sorted(hist['股票代码'].unique())
    seq=np.zeros((1,len(sids),SEQ,n_feat),dtype=np.float32);varr=np.zeros(len(sids),dtype=bool)
    for i,sid in enumerate(sids):
        sd=hist[hist['股票代码']==sid].sort_values('日期')
        if len(sd)>=SEQ: seq[0,i]=sd[fcols].values[-SEQ:].astype(np.float32);varr[i]=True
    seq_t=torch.FloatTensor(seq).to(device)
    sc=mc_infer(models,seq_t)

    raw_hist=processed_raw[processed_raw['日期']<=pd_str];sl=list(sids)
    filt=volatility_filter(raw_hist,sl,pd_str,top_pct=VP)
    bnc=bounce_confirm(raw_hist,filt,pd_str,threshold=BT)
    qual=compute_quality_score(raw_hist,filt,pd_str)
    final={}
    for i,sid in enumerate(sids):
        if not varr[i] or sid not in filt: continue
        s=float(sc[i])
        if sid not in bnc: s*=BP
        s+=(qual.get(sid,0.5)-0.5)*QC;final[sid]=s
    top10=sorted(final.items(),key=lambda x:x[1],reverse=True)[:10]
    top5=top10[:5]
    picks=list(zip(*equal_weight_allocate([s for s,_ in top5])))
    t1=next(d for d in test_dates if d>=pd.to_datetime(pd_str))
    t5r=[d for d in test_dates if d>=pd.to_datetime(pd_str)];t5=t5r[min(4,len(t5r)-1)]
    ret=0.0
    for sid,w in picks:
        r1=test_df[(test_df['股票代码']==sid)&(test_df['日期']==t1)]
        r5=test_df[(test_df['股票代码']==sid)&(test_df['日期']==t5)]
        if len(r1)>0 and len(r5)>0:
            ret+=(float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])*w
    return ret,[s for s,_ in top5],top10

print('Loading models...')
H=load_m('model/stock_emb_8_hybrid',197)
A=load_m('model/stock_emb_8_alpha158',len(alpha_f))
p_h=preproc('model/stock_emb_8_hybrid')
p_a=preproc('model/stock_emb_8_alpha158')

all_data={}
for pd_str in ['2026-06-01','2026-06-08']:
    data=processed_raw[processed_raw['日期']<=pd_str]
    recent=data[data['日期']>=pd.to_datetime(pd_str)-pd.Timedelta(days=10)]
    daily_ret=recent.groupby('日期')['涨跌幅'].mean()
    up_pct=(daily_ret>0).mean();trend=daily_ret.mean()*100

    print(f'\n{'='*70}')
    print(f'{pd_str}  |  up_pct={up_pct:.1%}  trend={trend:+.2f}%')
    print(f'{"="*70}')

    rh,top5h,top10h=eval_model(p_h,pd_str,fcols_all,197,H,'Hybrid')
    ra,top5a,top10a=eval_model(p_a,pd_str,alpha_f,len(alpha_f),A,'Alpha')

    # Simple avg of scores
    # Need both scores on same sids — use hybrid sids as reference
    hist_h=p_h[p_h['日期']<=pd_str];sids_h=sorted(hist_h['股票代码'].unique())
    sc_h=mc_infer(H,torch.FloatTensor(np.zeros((1,len(sids_h),SEQ,197),dtype=np.float32)).to(device))
    # Actually reuse cached scores from eval_model... simpler: just compute combined
    # For display: show individual results
    print(f'Hybrid:    {top5h} => {rh*100:+.2f}%')
    print(f'  Top-10: {[(s,round(sc,4)) for s,sc in top10h]}')
    print(f'Alpha158:  {top5a} => {ra*100:+.2f}%')
    print(f'  Top-10: {[(s,round(sc,4)) for s,sc in top10a]}')

    # Overlap analysis
    h_set=set(s for s,_ in top10h)
    a_set=set(s for s,_ in top10a)
    overlap=h_set&a_set
    print(f'Top-10 overlap: {len(overlap)}/10 ({sorted(overlap)})')

    all_data[pd_str]={'h':rh,'a':ra,'h_picks':top5h,'a_picks':top5a}

print(f'\n{"="*70}')
print(f'SUMMARY')
print(f'{"="*70}')
ah1=all_data['2026-06-01']['a'];hh1=all_data['2026-06-01']['h']
ah2=all_data['2026-06-08']['a'];hh2=all_data['2026-06-08']['h']
print(f'              {"W1":>10} {"W2":>10} {"Avg":>10}')
print(f'Always Hybrid  {hh1*100:+8.2f}% {hh2*100:+8.2f}% {(hh1+hh2)/2*100:+8.2f}%')
print(f'Always Alpha   {ah1*100:+8.2f}% {ah2*100:+8.2f}% {(ah1+ah2)/2*100:+8.2f}%')
# Simple avg
print(f'Simple avg     {(ah1+hh1)/2*100:+8.2f}% {(ah2+hh2)/2*100:+8.2f}% {(ah1+hh1+ah2+hh2)/4*100:+8.2f}%')
print(f'\nDone!')
