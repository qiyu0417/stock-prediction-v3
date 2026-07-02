"""Evaluate Hybrid & Alpha158 for 3 weeks in June"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed, _ALPHA_158_COLS
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

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

print('Loading models...')
import ensemble_models as _em
_orig_ti=_em.StockTransformerExpert.__init__;_orig_ci=_em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert,ConvStockExpert

def load_m(mdir,nf):
    _em.StockTransformerExpert.__init__=_orig_ti;_em.ConvStockExpert.__init__=_orig_ci
    with open(os.path.join(mdir,'ensemble_config.json')) as f: cfg=json.load(f)
    emb=cfg.get('stock_embed_dim',8)
    with open('model/stock_emb_8_hybrid/ensemble_config.json') as f2: ns=json.load(f2)['num_stocks']
    models=[]
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

H=load_m('model/stock_emb_8_hybrid',197);p_h=preproc('model/stock_emb_8_hybrid')
A=load_m('model/stock_emb_8_alpha158',len(alpha_f));p_a=preproc('model/stock_emb_8_alpha158')

# 三周预测日期
# W1: 6/1-6/5, 预测5/29(周五), T+1=6/1 open, T+5=6/5 open
# W2: 6/8-6/12, 预测6/5(周五), T+1=6/8 open, T+5=6/12 open
# W3: 6/15-6/18(端午19号放假,仅4天), 预测6/12(周五), T+1=6/15 open, T+4=6/18 open
weeks=[
    ('W1 (6/1-6/5)',  pd.to_datetime('2026-05-29'), pd.to_datetime('2026-06-01'), pd.to_datetime('2026-06-05'), 5),
    ('W2 (6/8-6/12)', pd.to_datetime('2026-06-05'), pd.to_datetime('2026-06-08'), pd.to_datetime('2026-06-12'), 5),
    ('W3 (6/15-6/18,4d)', pd.to_datetime('2026-06-12'), pd.to_datetime('2026-06-15'), pd.to_datetime('2026-06-18'), 4),
]

all_results=[]
for wname, pd_str, t1_date, t5_date, ndays in weeks:
    # Market context
    recent=processed_raw[(processed_raw['日期']<=pd_str)&(processed_raw['日期']>=pd_str-pd.Timedelta(days=10))]
    daily_ret=recent.groupby('日期')['涨跌幅'].mean()
    up_pct=(daily_ret>0).mean();trend=daily_ret.mean()*100

    for label, M, p, fcols, nf in [
        ('Hybrid', H, p_h, fcols_all, 197),
        ('Alpha158', A, p_a, alpha_f, len(alpha_f)),
    ]:
        hist=p[p['日期']<=pd_str];sids=sorted(hist['股票代码'].unique())
        seq=np.zeros((1,len(sids),SEQ,nf),dtype=np.float32);varr=np.zeros(len(sids),dtype=bool)
        for i,sid in enumerate(sids):
            sd=hist[hist['股票代码']==sid].sort_values('日期')
            if len(sd)>=SEQ: seq[0,i]=sd[fcols].values[-SEQ:].astype(np.float32);varr[i]=True

        seq_t=torch.FloatTensor(seq).to(device)
        all_s=[]
        for _ in range(MC):
            ps=[]
            for m in M:
                with torch.no_grad():
                    out=m(seq_t)
                    if isinstance(out,tuple): out=out[0]
                    ps.append(out[0].cpu().numpy())
            all_s.append(np.mean(ps,axis=0))
        sc=np.mean(all_s,axis=0)

        raw_hist=processed_raw[processed_raw['日期']<=pd_str];sl=list(sids)
        filt=volatility_filter(raw_hist,sl,str(pd_str.date()),top_pct=VP)
        bnc=bounce_confirm(raw_hist,filt,str(pd_str.date()),threshold=BT)
        qual=compute_quality_score(raw_hist,filt,str(pd_str.date()))
        final={}
        for i,sid in enumerate(sids):
            if not varr[i] or sid not in filt: continue
            s=float(sc[i])
            if sid not in bnc: s*=BP
            s+=(qual.get(sid,0.5)-0.5)*QC;final[sid]=s
        top5=sorted(final.items(),key=lambda x:x[1],reverse=True)[:5]

        # Compute 5-day return
        _,weights=equal_weight_allocate([s for s,_ in top5])
        ret=0.0;details=[]
        all_data=pd.concat([test_df,new_df])
        for sid,w in zip([s for s,_ in top5],weights):
            r1=all_data[(all_data['股票代码']==sid)&(all_data['日期']==t1_date)]
            r5=all_data[(all_data['股票代码']==sid)&(all_data['日期']==t5_date)]
            if len(r1)>0 and len(r5)>0:
                sr=(float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])
                ret+=sr*w
                details.append(f'{sid}:{sr*100:+.1f}%')
            else:
                details.append(f'{sid}:NODATA')

        all_results.append({'Week':wname,'Model':label,'Return':ret,'Top5':','.join([s for s,_ in top5]),'Details':' | '.join(details)})
        print(f'{wname} {label}: {ret*100:+.2f}% ({ndays}d)  Top-5: {[s for s,_ in top5]}')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
df=pd.DataFrame(all_results)
pvt=df.pivot(index='Week',columns='Model',values='Return')
pvt=pvt*100
print(pvt.to_string(float_format=lambda x: f'{x:+.2f}%'))
print(f"\nAvg: Hybrid={df[df['Model']=='Hybrid']['Return'].mean()*100:+.2f}%  Alpha158={df[df['Model']=='Alpha158']['Return'].mean()*100:+.2f}%")
