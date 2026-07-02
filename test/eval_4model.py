"""4-model blend: GNN + Hybrid + Alpha158 + EMA/ListMLE. MC=20 + 5 seeds."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, joblib
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC, SEQ = 20, 60
device = torch.device('cuda')
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

# ── Load data ──
print("Loading data...")
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str}); train_df['股票代码']=train_df['股票代码'].astype(str).str.zfill(6); train_df['日期']=pd.to_datetime(train_df['日期'],format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str}); test_df['股票代码']=test_df['股票代码'].astype(str).str.zfill(6); test_df['日期']=pd.to_datetime(test_df['日期'],format='mixed')
new_df = pd.read_csv('data/new_week.csv', dtype={'股票代码': str}); new_df['股票代码']=new_df['股票代码'].astype(str).str.zfill(6); new_df['日期']=pd.to_datetime(new_df['日期'],format='mixed')
full_df = pd.concat([train_df,test_df,new_df]).drop_duplicates(subset=['股票代码','日期'],keep='last')
full_df = full_df.sort_values(['股票代码','日期']).reset_index(drop=True)
all_sids = sorted(full_df['股票代码'].unique()); sid2idx = {s:i for i,s in enumerate(all_sids)}

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]; fcols_all = feature_cloums_map[FEATURE_NUM]
groups = [g.reset_index(drop=True) for _,g in full_df.groupby('股票代码',sort=False) if len(g)>=SEQ+10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument']=processed_raw['股票代码'].map(sid2idx)
processed_raw=processed_raw.dropna(subset=['instrument']).copy(); processed_raw['instrument']=processed_raw['instrument'].astype(np.int64)
processed_raw=_build_label_and_clean(processed_raw,drop_small_open=True)
alpha_f=[f for f in _ALPHA_158_COLS if f in fcols_all]

# ── GNN module and monkey-patch ──
import ensemble_models as _em
_orig_ti, _orig_ci = _em.StockTransformerExpert.__init__, _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

class GraphStockConv(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear=torch.nn.Linear(d_model,d_model); self.norm=torch.nn.LayerNorm(d_model)
        self.dropout=torch.nn.Dropout(dropout)
    def set_adjacency(self, adj_norm): self.register_buffer('adj_norm', adj_norm)
    def forward(self, x):
        B,N,D=x.shape; adj=self.adj_norm[:N,:N].to(x.device)
        o=torch.bmm(adj.unsqueeze(0).expand(B,-1,-1), self.linear(x))
        return self.norm(x+self.dropout(torch.nn.functional.relu(o)))

def build_adj(stock_ids):
    ind=pd.read_csv('data/industry.csv',dtype={'股票代码':str}); ind['股票代码']=ind['股票代码'].astype(str).str.zfill(6)
    sm={}; [sm.update({r['股票代码']:r.get('sector','Z')}) for _,r in ind.iterrows()]
    N=len(stock_ids); adj=np.zeros((N,N),dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if sm.get(stock_ids[i],'Z')==sm.get(stock_ids[j],'Z'): adj[i,j]=1.0
    adj+=np.eye(N,dtype=np.float32); deg=adj.sum(1)
    return torch.FloatTensor(np.diag(1/np.sqrt(np.maximum(deg,1e-8)))@adj@np.diag(1/np.sqrt(np.maximum(deg,1e-8))))

_gnn_adj=build_adj(all_sids)

def _gnn_trans_init(s,i,c,n): _orig_ti(s,i,c,n); g=GraphStockConv(s.d_model,c.get('dropout',0.1)); g.set_adjacency(_gnn_adj); s.cross_stock_attention=g
def _gnn_conv_init(s,i,c,n): _orig_ci(s,i,c,n); d=c.get('hidden_channels',256); g=GraphStockConv(d,c.get('dropout',0.1)); g.set_adjacency(_gnn_adj); s.cross_stock_attention=g

def load_model(mdir, nf, gnn=False):
    if gnn:
        _em.StockTransformerExpert.__init__=_gnn_trans_init; _em.ConvStockExpert.__init__=_gnn_conv_init
        StockTransformerExpert.__init__=_gnn_trans_init; ConvStockExpert.__init__=_gnn_conv_init
    else:
        _em.StockTransformerExpert.__init__=_orig_ti; _em.ConvStockExpert.__init__=_orig_ci
        StockTransformerExpert.__init__=_orig_ti; ConvStockExpert.__init__=_orig_ci
    with open(os.path.join(mdir,'ensemble_config.json')) as f: cfg=json.load(f)
    emb=cfg.get('stock_embed_dim',8)
    with open('model/stock_emb_8_hybrid/ensemble_config.json') as f2: ns=json.load(f2)['num_stocks']
    models=[]
    for ec in cfg['expert_configs']:
        path=os.path.join(mdir,f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        e=dict(ec); e['stock_embed_dim']=emb
        m=StockTransformerExpert(nf,e,ns) if ec['type']=='transformer' else ConvStockExpert(nf,e,ns)
        m.load_state_dict(torch.load(path,map_location=device,weights_only=True)); m.to(device); m.train()
        models.append(m)
    return models

def preprocess(mdir):
    with open(os.path.join(mdir,'winsor_bounds.json')) as f: wb=json.load(f)
    sc=joblib.load(os.path.join(mdir,'scaler.pkl')); p=processed_raw.copy()
    p[fcols_all]=p[fcols_all].replace([np.inf,-np.inf],np.nan).dropna(subset=fcols_all)
    for col,(lo,hi) in wb.items():
        if col in p.columns: p[col]=p[col].clip(lo,hi)
    p[fcols_all]=sc.transform(p[fcols_all]); return p

# ── Load all models ──
print("Loading models...")
G=load_model('model/stock_emb_8_gnn', len(fcols_all), gnn=True); pG=preprocess('model/stock_emb_8_gnn')
H=load_model('model/stock_emb_8_hybrid', len(fcols_all)); pH=preprocess('model/stock_emb_8_hybrid')
A=load_model('model/stock_emb_8_alpha158', len(alpha_f)); pA=preprocess('model/stock_emb_8_alpha158')
E=load_model('model/stock_emb_8_ema', len(fcols_all)); pE=preprocess('model/stock_emb_8_ema')
L=load_model('model/stock_emb_8_listmle_k3_t0.5', len(fcols_all)); pL=preprocess('model/stock_emb_8_listmle_k3_t0.5')
print(f"GNN={len(G)} H={len(H)} A={len(A)} EMA={len(E)} ListMLE={len(L)}")

def build_seq(p,ref_date,fcols,nf):
    hist=p[p['日期']<=ref_date]; sids=sorted(hist['股票代码'].unique())
    seq=np.zeros((1,len(sids),SEQ,nf),dtype=np.float32); valid=np.zeros(len(sids),dtype=bool)
    for i,sid in enumerate(sids):
        sd=hist[hist['股票代码']==sid].sort_values('日期')
        if len(sd)>=SEQ: seq[0,i]=sd[fcols].values[-SEQ:].astype(np.float32); valid[i]=True
    return seq,sids,valid

def mc_infer(models, seq_t, n=MC):
    all_s=[]
    for _ in range(n):
        ps=[]
        for m in models:
            with torch.no_grad():
                o=m(seq_t)
                if isinstance(o,tuple): o=o[0]
                ps.append(o[0].cpu().numpy())
        all_s.append(np.mean(ps,axis=0))
    return np.mean(all_s,axis=0)

def top5_ret(sids_top, t1_date, t5_date):
    _,w=equal_weight_allocate(sids_top); ret=0.0
    for sid,wgt in zip(sids_top,w):
        r1=full_df[(full_df['股票代码']==sid)&(full_df['日期']==t1_date)]
        r5=full_df[(full_df['股票代码']==sid)&(full_df['日期']==t5_date)]
        if len(r1)>0 and len(r5)>0: ret+=(float(r5.iloc[0]['开盘'])-float(r1.iloc[0]['开盘']))/float(r1.iloc[0]['开盘'])*wgt
    return ret

# ═══════════════════════════════════════════════════
weeks=[('W1',pd.to_datetime('2026-05-29'),pd.to_datetime('2026-06-01'),pd.to_datetime('2026-06-05'),5),
       ('W2',pd.to_datetime('2026-06-05'),pd.to_datetime('2026-06-08'),pd.to_datetime('2026-06-12'),5),
       ('W3',pd.to_datetime('2026-06-12'),pd.to_datetime('2026-06-15'),pd.to_datetime('2026-06-18'),4)]
seeds=[42,123,456,789,1024]

# Strategy weights: G, H, A, E/M
strategies=[
    ('G+H+A (0.15/0.45/0.4)',  0.15, 0.45, 0.4, 0.0, 0.0),
    # Add EMA
    ('+EMA 0.1',  0.1, 0.4, 0.4, 0.1, 0.0),
    ('+EMA 0.15', 0.1, 0.4, 0.35, 0.15, 0.0),
    ('+EMA 0.2',  0.1, 0.35, 0.35, 0.2, 0.0),
    # Add ListMLE
    ('+ListMLE 0.1',  0.1, 0.4, 0.4, 0.0, 0.1),
    ('+ListMLE 0.15', 0.1, 0.4, 0.35, 0.0, 0.15),
    ('+ListMLE 0.2',  0.1, 0.35, 0.35, 0.0, 0.2),
    # Both
    ('+E+L (0.1+0.1)', 0.1, 0.35, 0.35, 0.1, 0.1),
    ('+E+L (0.15+0.1)', 0.1, 0.35, 0.3, 0.15, 0.1),
]

all_results=[]

for seed in seeds:
    set_seed(seed); print(f"\n--- Seed={seed} ---")
    for wname,pd_str,t1_date,t5_date,ndays in weeks:
        ref_date=pd_str; ref_str=str(pd_str.date())
        seqG,sidsG,validG=build_seq(pG,ref_date,fcols_all,len(fcols_all))
        seqH,sidsH,validH=build_seq(pH,ref_date,fcols_all,len(fcols_all))
        seqA,sidsA,validA=build_seq(pA,ref_date,alpha_f,len(alpha_f))
        seqE,sidsE,validE=build_seq(pE,ref_date,fcols_all,len(fcols_all))
        seqL,sidsL,validL=build_seq(pL,ref_date,fcols_all,len(fcols_all))

        rG=mc_infer(G,torch.FloatTensor(seqG).to(device))
        rH=mc_infer(H,torch.FloatTensor(seqH).to(device))
        rA=mc_infer(A,torch.FloatTensor(seqA).to(device))
        rE=mc_infer(E,torch.FloatTensor(seqE).to(device))
        rL=mc_infer(L,torch.FloatTensor(seqL).to(device))

        raw_hist=processed_raw[processed_raw['日期']<=ref_date]

        for strat_name,wg,wh,wa,we,wl in strategies:
            gM={s:float(rG[i]) for i,s in enumerate(sidsG) if validG[i]}
            hM={s:float(rH[i]) for i,s in enumerate(sidsH) if validH[i]}
            aM={s:float(rA[i]) for i,s in enumerate(sidsA) if validA[i]}
            eM={s:float(rE[i]) for i,s in enumerate(sidsE) if validE[i]}
            lM={s:float(rL[i]) for i,s in enumerate(sidsL) if validL[i]}
            common=sorted(set(gM)&set(hM)&set(aM))
            if we>0: common=sorted(set(common)&set(eM))
            if wl>0: common=sorted(set(common)&set(lM))
            if len(common)<10: continue

            combined={}
            for sid in common:
                s=wg*gM.get(sid,0)+wh*hM.get(sid,0)+wa*aM.get(sid,0)
                if we>0: s+=we*eM.get(sid,0)
                if wl>0: s+=wl*lM.get(sid,0)
                combined[sid]=s

            filt=volatility_filter(raw_hist,list(common),ref_str,top_pct=VP)
            if len(filt)<5: continue
            bnc=bounce_confirm(raw_hist,filt,ref_str,threshold=BT)
            qual=compute_quality_score(raw_hist,filt,ref_str)
            final={}
            for sid in filt:
                s=combined.get(sid,-999)
                if sid not in bnc: s*=BP
                s+=(qual.get(sid,0.5)-0.5)*QC; final[sid]=s
            top5=sorted(final.items(),key=lambda x:x[1],reverse=True)[:5]
            ret=top5_ret([s for s,_ in top5],t1_date,t5_date)
            all_results.append({'Seed':seed,'Week':wname,'Strategy':strat_name,'Return':ret,
                               'Top5':','.join([s for s,_ in top5])})
        print(f"  {wname} ok")

# ── Summary ──
df=pd.DataFrame(all_results)
print(f"\n{'='*70}")
print("RESULTS: MC=20 5-seed mean returns")
print(f"{'='*70}")
for strat_name,_,_,_,_,_ in strategies:
    sub=df[df['Strategy']==strat_name]
    if len(sub)>0: print(f"  {strat_name:30s}: {sub['Return'].mean()*100:+.2f}% ±{sub['Return'].std()*100:.2f}%")

print(f"\n{'='*70}")
print("WEEKLY BREAKDOWN")
print(f"{'='*70}")
for week_name,_,_,_,_ in weeks:
    print(f"\n  {week_name}:")
    for strat_name,_,_,_,_,_ in strategies:
        sub=df[(df['Week']==week_name)&(df['Strategy']==strat_name)]
        if len(sub)>0: print(f"    {strat_name:30s}: {sub['Return'].mean()*100:+.2f}% ±{sub['Return'].std()*100:.2f}%")
print(f"\nDone!")
