"""V6 vs V7 comparison"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
import gc

from config_v5 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from market_regime import compute_market_regime
from quality_filter import (
    bounce_confirm, compute_quality_score,
    confidence_weighted_allocate, volatility_filter
)

TRAIN = 'data/train.csv'
V5D = 'model/v5_ensemble'
V7D = 'model/v7_ensemble'

MONTHS = {
    '2026-01': ('2025-12-31', ['2026-01-02','2026-01-05','2026-01-06','2026-01-07','2026-01-08']),
    '2026-02': ('2026-01-27', ['2026-02-02','2026-02-03','2026-02-04','2026-02-05','2026-02-06']),
    '2026-03': ('2026-02-27', ['2026-03-02','2026-03-03','2026-03-04','2026-03-05','2026-03-06']),
    '2026-04': ('2026-03-31', ['2026-04-01','2026-04-02','2026-04-03','2026-04-07','2026-04-08']),
    '2026-05': ('2026-04-30', ['2026-05-04','2026-05-05','2026-05-06','2026-05-07','2026-05-08']),
}


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def pp(df, sid2idx, winsor, scaler):
    fe = _engineer_158plus39
    fc = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码','日期']).reset_index(drop=True)
    groups = [g for _,g in df.groupby('股票代码',sort=False)]
    p = pd.concat([fe(g) for g in tqdm(groups, desc='FE', leave=False)]).reset_index(drop=True)
    p['instrument'] = p['股票代码'].map(sid2idx)
    p = p.dropna(subset=['instrument']).copy()
    p['instrument'] = p['instrument'].astype(np.int64)
    p['日期'] = pd.to_datetime(p['日期'])
    for col,(lo,hi) in winsor.items():
        if col in p.columns: p[col] = p[col].clip(lo,hi)
    common = [c for c in scaler.feature_names_in_ if c in p.columns]
    p[common] = scaler.transform(p[common])
    return p, common


def bs(data, features, stock_ids, target_date):
    seqs, sids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码']==sid)&(data['日期']<=target_date)].sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist)==SEQUENCE_LENGTH:
            seqs.append(hist[features].values.astype(np.float32))
            sids.append(sid)
    return np.asarray(seqs, dtype=np.float32) if seqs else np.array([]), sids


def lm(model_dir, fdim, nstocks, device):
    with open(os.path.join(model_dir, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg=json.load(f)
    models=[]
    for ec in cfg['expert_configs']:
        path=os.path.join(model_dir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        if ec['type']=='transformer': m=StockTransformerExpert(fdim, ec, nstocks)
        elif ec['type']=='conv': m=ConvStockExpert(fdim, ec, nstocks)
        else: continue
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device); models.append(m)
    return models, [1.0/len(models)]*len(models)


def v6_predict(experts, wts, x, device, seq_ids, proc_data, all_stocks, ref_date):
    cs = MAX_STOCKS_PER_CHUNK if device.type=='cuda' else 9999
    ua = USE_AMP and device.type=='cuda'
    all_f = []
    for r in range(5):
        torch.manual_seed(42+r*100); np.random.seed(42+r*100)
        rnd=[]
        for expert in experts:
            expert.train(); mc=[]
            with torch.no_grad():
                for _ in range(30):
                    with torch.amp.autocast('cuda', enabled=ua):
                        if x.size(1)<=cs: s=expert(x).squeeze(0)
                        else:
                            cs_list=[]
                            for start in range(0,x.size(1),cs):
                                end=min(start+cs,x.size(1))
                                cs_list.append(expert(x[:,start:end].contiguous()).squeeze(0))
                            s=torch.cat(cs_list,dim=0)
                    mc.append(s)
            rnd.append(torch.stack(mc).mean(dim=0).cpu().numpy())
        fused=np.zeros(len(rnd[0]))
        for w,sc in zip(wts,rnd): fused += w*sc
        all_f.append(fused)
    raw=np.mean(all_f,axis=0)
    score_map={sid:float(raw[i]) for i,sid in enumerate(seq_ids) if i<len(raw)}
    regime=compute_market_regime(proc_data,[],all_stocks,pd.to_datetime(ref_date))
    if regime.get('skip_trading'): return [],[]
    fids=volatility_filter(proc_data,seq_ids,ref_date,top_pct=0.95)
    if len(fids)<3: fids=seq_ids[:10]
    confirmed=bounce_confirm(proc_data,fids,ref_date)
    for sid in fids:
        if sid not in confirmed: score_map[sid]=score_map.get(sid,0)*0.92
    quality=compute_quality_score(proc_data,fids,ref_date)
    for sid in fids:
        if sid in score_map and sid in quality: score_map[sid]+=(quality[sid]-0.5)*0.05
    sorted_s=sorted(fids, key=lambda s:score_map.get(s,-999), reverse=True)
    sel,w=confidence_weighted_allocate(score_map,sorted_s,{},max_positions=5,temperature=0.3,max_single=0.30,use_sigma=False)
    return sel,w


def calc_ret(stocks, wts, data, dates):
    wd=data[data['日期'].isin(pd.to_datetime(dates))]
    f=wd[wd['股票代码'].isin(stocks)]
    if f.empty: return 0.0
    t=0.0
    for sid,w in zip(stocks,wts):
        sw=f[f['股票代码']==sid].sort_values('日期')
        if len(sw)>=2: t+=w*(sw.iloc[-1]['开盘']-sw.iloc[0]['开盘'])/sw.iloc[0]['开盘']
    return t


def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    full=pd.read_csv(TRAIN,dtype={'股票代码':str})
    full['股票代码']=full['股票代码'].astype(str).str.zfill(6)
    full['日期']=pd.to_datetime(full['日期'])
    all_stocks=sorted(full['股票代码'].unique())
    fdim=len(feature_cloums_map[FEATURE_NUM])
    nstocks=len(all_stocks)

    sv5=joblib.load(os.path.join(V5D,'scaler.pkl'))
    with open(os.path.join(V5D,'winsor_bounds.json')) as f: wv5=json.load(f)
    v5m,v5w=lm(V5D,fdim,nstocks,device)

    sv7=joblib.load(os.path.join(V7D,'scaler.pkl'))
    with open(os.path.join(V7D,'winsor_bounds.json')) as f: wv7=json.load(f)
    v7m,v7w=lm(V7D,fdim,nstocks,device)

    print(f"V5: {len(v5m)} experts, V7: {len(v7m)} experts\n")

    rs=[]
    for month,(cutoff,wdates) in MONTHS.items():
        print(f"{'='*60}")
        print(f"  {month} | cutoff={cutoff}")
        train_df=full[full['日期']<=cutoff].copy()
        sid2idx={s:i for i,s in enumerate(all_stocks)}

        pv5,cv5=pp(train_df,sid2idx,wv5,sv5)
        sv5s,ids5=bs(pv5,cv5,all_stocks,pd.to_datetime(cutoff))
        xv5=torch.from_numpy(sv5s).unsqueeze(0).to(device)

        pv7,cv7=pp(train_df,sid2idx,wv7,sv7)
        sv7s,ids7=bs(pv7,cv7,all_stocks,pd.to_datetime(cutoff))
        xv7=torch.from_numpy(sv7s).unsqueeze(0).to(device)

        v6s,v6w=v6_predict(v5m,v5w,xv5,device,ids5,pv5,all_stocks,cutoff)
        v7s,v7w=v6_predict(v7m,v7w,xv7,device,ids7,pv7,all_stocks,cutoff)

        r6=calc_ret(v6s,v6w,full,wdates)
        r7=calc_ret(v7s,v7w,full,wdates)

        print(f"  V6: {v6s} -> {r6:+.4%}")
        print(f"  V7: {v7s} -> {r7:+.4%}")
        d=r7-r6
        flag='V7' if d>0.001 else ('V6' if d<-0.001 else '=')
        print(f"  {flag} ({d:+.4%})\n")

        rs.append({'m':month,'r6':r6,'r7':r7,'d':d})
        del train_df,pv5,pv7; gc.collect()

    print("="*60)
    print(f"  {'Month':<8} {'V6':>8} {'V7':>8} {'Diff':>8}")
    t6=t7=0
    for r in rs:
        t6+=r['r6'];t7+=r['r7']
        print(f"  {r['m']:<8} {r['r6']:>+8.4%} {r['r7']:>+8.4%} {r['d']:>+8.4%}")
    print(f"  {'累计':<8} {t6:>+8.4%} {t7:>+8.4%}")


if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    main()
