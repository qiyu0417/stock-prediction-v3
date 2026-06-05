"""
组员 v1_ensemble (6专家+MetaAggregator) + V6管线 vs 我们V5/V6
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
from collections import Counter
import gc

# Load teammate's ensemble_models (has MonthSeasonalExpert) AFTER ours
# by importing only what's needed from their path
from config_v5 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert, MonthSeasonalExpert, MetaAggregator
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from market_regime import compute_market_regime
from quality_filter import (
    bounce_confirm, compute_quality_score,
    confidence_weighted_allocate, volatility_filter
)

TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
V5_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'v5_ensemble')
TM_ENSEMBLE = "C:/Users/73065/Desktop/gupiao-3-model/gupiao-3-model/model/v1_ensemble"

MONTHS = {
    '2026-01': ('2025-12-31', ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']),
    '2026-02': ('2026-01-27', ['2026-02-02', '2026-02-03', '2026-02-04', '2026-02-05', '2026-02-06']),
    '2026-03': ('2026-02-27', ['2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06']),
    '2026-04': ('2026-03-31', ['2026-04-01', '2026-04-02', '2026-04-03', '2026-04-07', '2026-04-08']),
    '2026-05': ('2026-04-30', ['2026-05-04', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08']),
}


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def preprocess_v5(df, stockid2idx, winsor_bounds, scaler):
    fe = feature_engineer_func_map[FEATURE_NUM]
    fcols = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    processed = pd.concat([fe(g) for g in tqdm(groups, desc='FE', leave=False)]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])
    for col, (lo, hi) in winsor_bounds.items():
        if col in processed.columns:
            processed[col] = processed[col].clip(lo, hi)
    common = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[common] = scaler.transform(processed[common])
    return processed, common


def build_sequences(data, features, stock_ids, target_date):
    sequences, seq_stock_ids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= target_date)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            sequences.append(hist[features].values.astype(np.float32))
            seq_stock_ids.append(sid)
    return np.asarray(sequences, dtype=np.float32) if sequences else np.array([]), seq_stock_ids


def load_v5(fdim, nstocks, device):
    with open(os.path.join(V5_DIR, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(V5_DIR, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        if ec['type'] == 'transformer': m = StockTransformerExpert(fdim, ec, nstocks)
        elif ec['type'] == 'conv': m = ConvStockExpert(fdim, ec, nstocks)
        else: continue
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device); models.append(m)
    return models, [1.0/len(models)]*len(models)


def load_tm_ensemble(fdim, nstocks, device):
    with open(os.path.join(TM_ENSEMBLE, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    experts, meta = [], None
    for ec in cfg['expert_configs']:
        path = os.path.join(TM_ENSEMBLE, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path): continue
        t = ec.get('type','transformer')
        if t == 'transformer': m = StockTransformerExpert(fdim, ec, nstocks)
        elif t == 'conv': m = ConvStockExpert(fdim, ec, nstocks)
        elif t == 'month_seasonal': m = MonthSeasonalExpert(fdim, ec, nstocks)
        elif t == 'adversarial': continue
        else: continue
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device); experts.append(m)

    meta_path = os.path.join(TM_ENSEMBLE, 'meta_aggregator.pth')
    if os.path.exists(meta_path):
        meta = MetaAggregator(len(experts), nstocks, hidden_dim=64).to(device)
        meta.load_state_dict(torch.load(meta_path, map_location=device))
        meta.eval()
    return experts, meta


def tm_predict(experts, meta, x, device):
    """Copy of their predict_smart.py logic: MC -> MetaAggregator"""
    mc_samples = 20
    all_scores = []
    for e in experts:
        e.train()
        mc = []
        with torch.no_grad():
            for _ in range(mc_samples):
                mc.append(e(x).squeeze(0))
        all_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())

    if meta is not None:
        es = torch.from_numpy(np.stack(all_scores, axis=-1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            final = meta(es).squeeze(0).cpu().numpy()
    else:
        final = np.mean(all_scores, axis=0)
    return final


def v5_predict(experts, weights, x, device, seq_ids, risk_scores, stress):
    NUM_ROUNDS, MC_SPR = 5, 30
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'
    from risk_filter import apply_risk_filter

    all_top5 = []
    for r in range(NUM_ROUNDS):
        torch.manual_seed(42+r*100); np.random.seed(42+r*100)
        rnd_scores = []
        for expert in experts:
            expert.train(); mc = []
            with torch.no_grad():
                for _ in range(MC_SPR):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x.size(1) <= chunk_size: s = expert(x).squeeze(0)
                        else:
                            cs = []
                            for start in range(0, x.size(1), chunk_size):
                                end = min(start+chunk_size, x.size(1))
                                cs.append(expert(x[:, start:end].contiguous()).squeeze(0))
                            s = torch.cat(cs, dim=0)
                    mc.append(s)
            rnd_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())
        fused = np.zeros(len(rnd_scores[0]))
        for w, sc in zip(weights, rnd_scores): fused += w*sc
        sel, _ = apply_risk_filter(fused, seq_ids, risk_scores, stress, max_risk_score=85, min_positions=3, max_positions=5)
        all_top5.extend(sel)
    vc = Counter(all_top5)
    consensus = [s for s,c in vc.most_common() if c>=3]
    if not consensus: consensus = [s for s,_ in vc.most_common(3)]
    if len(consensus)>5: consensus=consensus[:5]
    return consensus, [1.0/len(consensus)]*len(consensus) if consensus else []


def v6_post(raw_scores, seq_ids, proc_data, all_stocks, ref_date):
    score_map = {sid: float(raw_scores[i]) for i,sid in enumerate(seq_ids) if i<len(raw_scores)}
    regime = compute_market_regime(proc_data, [], all_stocks, pd.to_datetime(ref_date))
    if regime.get('skip_trading'): return [], [], regime
    fids = volatility_filter(proc_data, seq_ids, ref_date, top_pct=0.95)
    if len(fids)<3: fids=seq_ids[:10]
    confirmed = bounce_confirm(proc_data, fids, ref_date)
    for sid in fids:
        if sid not in confirmed: score_map[sid]=score_map.get(sid,0)*0.92
    quality = compute_quality_score(proc_data, fids, ref_date)
    for sid in fids:
        if sid in score_map and sid in quality: score_map[sid]+=(quality[sid]-0.5)*0.05
    sorted_s = sorted(fids, key=lambda s:score_map.get(s,-999), reverse=True)
    sel, wts = confidence_weighted_allocate(score_map, sorted_s, {}, max_positions=5, temperature=0.3, max_single=0.30, use_sigma=False)
    return sel, wts, regime


def calc_ret(stocks, wts, data, dates):
    wd = data[data['日期'].isin(pd.to_datetime(dates))]
    f = wd[wd['股票代码'].isin(stocks)]
    if f.empty: return 0.0
    total = 0.0
    for sid,w in zip(stocks,wts):
        sw = f[f['股票代码']==sid].sort_values('日期')
        if len(sw)>=2: total += w*(sw.iloc[-1]['开盘']-sw.iloc[0]['开盘'])/sw.iloc[0]['开盘']
    return total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    full_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码':str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    all_stocks = sorted(full_df['股票代码'].unique())
    fdim = len(feature_cloums_map[FEATURE_NUM])
    nstocks = len(all_stocks)

    # Load V5
    scaler_v5 = joblib.load(os.path.join(V5_DIR, 'scaler.pkl'))
    with open(os.path.join(V5_DIR, 'winsor_bounds.json')) as f: winsor = json.load(f)
    v5_models, v5_w = load_v5(fdim, nstocks, device)

    # Load teammate ensemble + their scaler
    tm_scaler = joblib.load(os.path.join(TM_ENSEMBLE, 'scaler.pkl'))
    tm_experts, tm_meta = load_tm_ensemble(fdim, nstocks, device)
    print(f"V5: {len(v5_models)} experts, TM ensemble: {len(tm_experts)} experts + Meta\n")

    results = []
    for month, (cutoff, week_dates) in MONTHS.items():
        print(f"{'='*60}")
        print(f"  {month} | cutoff: {cutoff} | {week_dates[0]} ~ {week_dates[-1]}")
        train_df = full_df[full_df['日期']<=cutoff].copy()
        sid2idx = {s:i for i,s in enumerate(all_stocks)}

        # V5 preprocess
        pv5, cv5 = preprocess_v5(train_df, sid2idx, winsor, scaler_v5)
        sv5, ids5 = build_sequences(pv5, cv5, all_stocks, pd.to_datetime(cutoff))
        from risk_filter import compute_risk_scores
        rv5, sv = compute_risk_scores(pv5, cv5, all_stocks, ids5, train_df['日期'].max())
        xv5 = torch.from_numpy(sv5).unsqueeze(0).to(device)

        # TM preprocess (their scaler, no Winsorization)
        fe = feature_engineer_func_map[FEATURE_NUM]
        fc = feature_cloums_map[FEATURE_NUM]
        df_tm = train_df.sort_values(['股票代码','日期']).reset_index(drop=True)
        grps = [g for _,g in df_tm.groupby('股票代码',sort=False)]
        ptm = pd.concat([fe(g) for g in tqdm(grps, desc='TM FE', leave=False)]).reset_index(drop=True)
        ptm['instrument'] = ptm['股票代码'].map(sid2idx)
        ptm = ptm.dropna(subset=['instrument']).copy()
        ptm['instrument'] = ptm['instrument'].astype(np.int64)
        ptm['日期'] = pd.to_datetime(ptm['日期'])
        ptm[fc] = ptm[fc].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        ctm = [c for c in tm_scaler.feature_names_in_ if c in ptm.columns]
        ptm[ctm] = tm_scaler.transform(ptm[ctm])
        stm, ids_tm_s = build_sequences(ptm, ctm, all_stocks, pd.to_datetime(cutoff))
        xtm = torch.from_numpy(stm).unsqueeze(0).to(device)

        # V5 predict
        v5s, v5w = v5_predict(v5_models, v5_w, xv5, device, ids5, {s:rv5.get(s,50) for s in ids5}, sv)
        r5 = calc_ret(v5s, v5w, full_df, week_dates)

        # V6 predict
        v5_raw = _mc_raw(v5_models, v5_w, xv5, device)
        v6s, v6w, _ = v6_post(v5_raw, ids5, pv5, all_stocks, cutoff)
        r6 = calc_ret(v6s, v6w, full_df, week_dates)

        # TM predict
        tm_raw = tm_predict(tm_experts, tm_meta, xtm, device)
        tm_scores_sid = {ids_tm_s[i]:float(tm_raw[i]) for i in range(len(ids_tm_s))}

        # TM raw top5 equal
        tm_order = np.argsort(tm_raw)[::-1]
        tm_top = [ids_tm_s[i] for i in tm_order[:5] if i<len(ids_tm_s)]
        r_tm = calc_ret(tm_top, [0.2]*len(tm_top), full_df, week_dates)

        # TM + V6
        tm_v6s, tm_v6w, tm_regime = v6_post(tm_raw, ids_tm_s, ptm, all_stocks, cutoff)
        r_tmv6 = calc_ret(tm_v6s, tm_v6w, full_df, week_dates)

        print(f"  V5:    {v5s} → {r5:+.4%}")
        print(f"  V6:    {v6s} {[f'{w:.1%}' for w in v6w] if v6w else '空仓'} → {r6:+.4%}")
        print(f"  TM:    {tm_top} → {r_tm:+.4%}")
        print(f"  TM+V6: {tm_v6s} {[f'{w:.1%}' for w in tm_v6w] if tm_v6w else '空仓'} → {r_tmv6:+.4%} | {tm_regime['regime']}\n")

        results.append({'m':month,'r5':r5,'r6':r6,'rtm':r_tm,'rtmv6':r_tmv6})
        del train_df, pv5, ptm; gc.collect()

    print("="*60)
    print(f"  {'Month':<8} {'V5':>8} {'V6':>8} {'TM':>8} {'TM+V6':>8}")
    tv5=tv6=ttm=ttmv6=0
    for r in results:
        tv5+=r['r5'];tv6+=r['r6'];ttm+=r['rtm'];ttmv6+=r['rtmv6']
        print(f"  {r['m']:<8} {r['r5']:>+8.4%} {r['r6']:>+8.4%} {r['rtm']:>+8.4%} {r['rtmv6']:>+8.4%}")
    print(f"  {'累计':<8} {tv5:>+8.4%} {tv6:>+8.4%} {ttm:>+8.4%} {ttmv6:>+8.4%}")


def _mc_raw(experts, weights, x, device):
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type=='cuda' else 9999
    use_amp = USE_AMP and device.type=='cuda'
    all_fused = []
    for r in range(5):
        torch.manual_seed(42+r*100); np.random.seed(42+r*100)
        rnd = []
        for expert in experts:
            expert.train(); mc = []
            with torch.no_grad():
                for _ in range(30):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x.size(1)<=chunk_size: s=expert(x).squeeze(0)
                        else:
                            cs=[]
                            for start in range(0,x.size(1),chunk_size):
                                end=min(start+chunk_size,x.size(1))
                                cs.append(expert(x[:,start:end].contiguous()).squeeze(0))
                            s=torch.cat(cs,dim=0)
                    mc.append(s)
            rnd.append(torch.stack(mc).mean(dim=0).cpu().numpy())
        fused=np.zeros(len(rnd[0]))
        for w,sc in zip(weights,rnd): fused+=w*sc
        all_fused.append(fused)
    return np.mean(all_fused,axis=0)


if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    main()
