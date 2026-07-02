"""Minimal eval: add StatArb to existing 3-model blend. MC=20 + 5 seeds."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, joblib
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate
from config_stock_emb_8 import FEATURE_NUM
from ensemble_models import StockTransformerExpert, ConvStockExpert, StatArbRegressionExpert

MC, SEQ = 20, 60
device = torch.device('cuda')
VP, BT, BP, QC = 0.95, 0.008, 0.92, 0.05

print('Loading data...')
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6)
train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')
test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')
new_df = pd.read_csv('data/new_week.csv', dtype={'股票代码': str})
new_df['股票代码'] = new_df['股票代码'].astype(str).str.zfill(6)
new_df['日期'] = pd.to_datetime(new_df['日期'], format='mixed')
full_df = pd.concat([train_df, test_df, new_df]).drop_duplicates(subset=['股票代码', '日期'], keep='last')
full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
all_sids = sorted(full_df['股票代码'].unique())
sid2idx = {s: i for i, s in enumerate(all_sids)}

fe = feature_engineer_func_map[FEATURE_NUM]
fcols_all = feature_cloums_map[FEATURE_NUM]
groups = [g.reset_index(drop=True) for _, g in full_df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument'] = processed_raw['股票代码'].map(sid2idx)
processed_raw = processed_raw.dropna(subset=['instrument']).copy()
processed_raw['instrument'] = processed_raw['instrument'].astype(np.int64)
processed_raw = _build_label_and_clean(processed_raw, drop_small_open=True)
alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]

# ── Model loading ──
import ensemble_models as _em
_orig_ti, _orig_ci = _em.StockTransformerExpert.__init__, _em.ConvStockExpert.__init__


class GraphStockConv(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def set_adjacency(self, adj):
        self.register_buffer('adj_norm', adj)

    def forward(self, x):
        B, N, D = x.shape
        adj = self.adj_norm[:N, :N].to(x.device)
        o = torch.bmm(adj.unsqueeze(0).expand(B, -1, -1), self.linear(x))
        return self.norm(x + self.dropout(torch.nn.functional.relu(o)))


def build_adj(stock_ids):
    pvt = full_df.pivot(index='日期', columns='股票代码', values='涨跌幅')
    avail = [s for s in stock_ids if s in pvt.columns]
    pvt = pvt[avail].dropna()
    corr = pvt.corr().values
    M = len(avail)
    a2g = {s: stock_ids.index(s) for s in avail}
    N = len(stock_ids)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            if i != j and abs(corr[i, j]) > 0.5:
                adj[a2g[avail[i]], a2g[avail[j]]] = 1.0
    adj += np.eye(N, dtype=np.float32)
    deg = adj.sum(1)
    return torch.FloatTensor(np.diag(1 / np.sqrt(np.maximum(deg, 1e-8))) @ adj @ np.diag(
        1 / np.sqrt(np.maximum(deg, 1e-8))))


_gnn_adj = build_adj(all_sids)


def _gnn_init(cls, d_key='d_model'):
    def init_fn(s, i, c, n):
        _orig_ti(s, i, c, n) if cls == StockTransformerExpert else _orig_ci(s, i, c, n)
        d = c.get(d_key, 256)
        g = GraphStockConv(d, c.get('dropout', 0.1))
        g.set_adjacency(_gnn_adj)
        s.cross_stock_attention = g
    return init_fn


def load_models(mdir, nf, gnn=False, statarb=False):
    if gnn:
        _em.StockTransformerExpert.__init__ = _gnn_init(StockTransformerExpert)
        _em.ConvStockExpert.__init__ = _gnn_init(ConvStockExpert, 'hidden_channels')
        StockTransformerExpert.__init__ = _gnn_init(StockTransformerExpert)
        ConvStockExpert.__init__ = _gnn_init(ConvStockExpert, 'hidden_channels')
    else:
        _em.StockTransformerExpert.__init__ = _orig_ti
        _em.ConvStockExpert.__init__ = _orig_ci
        StockTransformerExpert.__init__ = _orig_ti
        ConvStockExpert.__init__ = _orig_ci

    with open(os.path.join(mdir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    with open('model/stock_emb_8_hybrid/ensemble_config.json') as f2:
        ns = json.load(f2)['num_stocks']
    ms = []
    for ec in cfg['expert_configs']:
        p = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(p):
            continue
        e = dict(ec)
        e['stock_embed_dim'] = emb
        if statarb:
            m = StatArbRegressionExpert(nf, e, ns)
        elif ec['type'] == 'transformer':
            m = StockTransformerExpert(nf, e, ns)
        else:
            m = ConvStockExpert(nf, e, ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        ms.append(m)
    return ms


def preproc(mdir, use_features=None):
    if use_features is None:
        use_features = fcols_all
    with open(os.path.join(mdir, 'winsor_bounds.json')) as f:
        wb = json.load(f)
    sc = joblib.load(os.path.join(mdir, 'scaler.pkl'))
    needed = use_features + ['日期', '股票代码']
    p = processed_raw[needed].copy()
    p[use_features] = p[use_features].replace([np.inf, -np.inf], np.nan).dropna(subset=use_features)
    for col, (lo, hi) in wb.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    p[use_features] = sc.transform(p[use_features].values)
    return p


def build_seq(p, ref, fcols, nf):
    hist = p[p['日期'] <= ref]
    sids = sorted(hist['股票代码'].unique())
    seq = np.zeros((1, len(sids), SEQ, nf), dtype=np.float32)
    v = np.zeros(len(sids), dtype=bool)
    for i, sid in enumerate(sids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ:
            seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32)
            v[i] = True
    return seq, sids, v


def mc_infer(ms, st, n=MC):
    if not ms:
        return None
    all_s = []
    for _ in range(n):
        ps = []
        for m in ms:
            with torch.no_grad():
                o = m(st)
                if isinstance(o, tuple):
                    o = o[0]
                ps.append(o[0].cpu().numpy())
        all_s.append(np.mean(ps, axis=0))
    return np.mean(all_s, axis=0)


# ── Preprocess one at a time to save memory ──
print('Loading GNN models...')
G = load_models('model/stock_emb_8_gnn_corr', len(fcols_all), gnn=True)
print('Loading Hybrid models...')
H = load_models('model/stock_emb_8_hybrid', len(fcols_all))
print('Loading Alpha158 models...')
A = load_models('model/stock_emb_8_alpha158', len(alpha_f))
print('Loading StatArb models...')
S = load_models('model/stock_emb_8_statarb', len(fcols_all), statarb=True)
print(f'  GNN:{len(G)} Hybrid:{len(H)} Alpha158:{len(A)} StatArb:{len(S)}')

print('Preprocessing data (one model at a time)...')
pG = preproc('model/stock_emb_8_gnn_corr')
print('  GNN done')
pH = preproc('model/stock_emb_8_hybrid')
print('  Hybrid done')
pA = preproc('model/stock_emb_8_alpha158')  # Alpha158 scaler was fit on all 197 features
print('  Alpha158 done')
pS = preproc('model/stock_emb_8_statarb')
print('  StatArb done')

# ── Eval ──
weeks = [
    (pd.to_datetime('2026-05-28'), pd.to_datetime('2026-06-01'), pd.to_datetime('2026-06-05'), 'W1'),
    (pd.to_datetime('2026-06-04'), pd.to_datetime('2026-06-08'), pd.to_datetime('2026-06-12'), 'W2'),
    (pd.to_datetime('2026-06-11'), pd.to_datetime('2026-06-15'), pd.to_datetime('2026-06-18'), 'W3'),
    (pd.to_datetime('2026-06-18'), pd.to_datetime('2026-06-22'), pd.to_datetime('2026-06-26'), 'W4'),
]

seeds = [42, 123, 456, 789, 1024]
results = []

for pd_date, t1, t5, label in weeks:
    print(f'\n{"=" * 60}')
    print(f'{label}: predict={pd_date.date()}, T+1={t1.date()}, T+5={t5.date()}')
    print(f'{"=" * 60}')

    seqG, sG, vG = build_seq(pG, pd_date, fcols_all, len(fcols_all))
    seqH, sH, vH = build_seq(pH, pd_date, fcols_all, len(fcols_all))
    seqA, sA, vA = build_seq(pA, pd_date, alpha_f, len(alpha_f))
    seqS, sS, vS = build_seq(pS, pd_date, fcols_all, len(fcols_all))

    for seed in seeds:
        set_seed(seed)
        rG = mc_infer(G, torch.FloatTensor(seqG).to(device))
        rH = mc_infer(H, torch.FloatTensor(seqH).to(device))
        rA = mc_infer(A, torch.FloatTensor(seqA).to(device))
        rS = mc_infer(S, torch.FloatTensor(seqS).to(device))

        raw_hist = processed_raw[processed_raw['日期'] <= pd_date]

        gM = {s: float(rG[i]) for i, s in enumerate(sG) if vG[i]} if rG is not None else {}
        hM = {s: float(rH[i]) for i, s in enumerate(sH) if vH[i]} if rH is not None else {}
        aM = {s: float(rA[i]) for i, s in enumerate(sA) if vA[i]} if rA is not None else {}
        sM = {s: float(rS[i]) for i, s in enumerate(sS) if vS[i]} if rS is not None else {}

        common = sorted(set(hM) & set(aM))
        common_gs = sorted(set(gM) & set(hM) & set(aM) & set(sM))

        strategies = [
            ('CorrG+H+A (0.1/0.4/0.5)', {'GNN': 0.1, 'Hybrid': 0.4, 'Alpha158': 0.5}, common_gs),
            ('CorrG+H+A+S (0.1/0.3/0.4/0.2)', {'GNN': 0.1, 'Hybrid': 0.3, 'Alpha158': 0.4, 'StatArb': 0.2}, common_gs),
            ('H+A+S (0.35/0.4/0.25)', {'Hybrid': 0.35, 'Alpha158': 0.4, 'StatArb': 0.25}, common),
            ('H+A (0.45/0.55)', {'Hybrid': 0.45, 'Alpha158': 0.55}, common),
            ('Hybrid only', {'Hybrid': 1.0}, common),
            ('StatArb only', {'StatArb': 1.0}, [s for s in sM if s in common]),
        ]

        for slabel, weights, cmn in strategies:
            if not cmn:
                continue
            combined = {s: sum(weights.get(m, 0) * {'GNN': gM, 'Hybrid': hM, 'Alpha158': aM, 'StatArb': sM}[m].get(s, 0)
                             for m in weights) for s in cmn}
            filt = volatility_filter(raw_hist, list(cmn), str(pd_date.date()), top_pct=VP)
            if len(filt) < 5:
                continue
            bnc = bounce_confirm(raw_hist, filt, str(pd_date.date()), threshold=BT)
            qual = compute_quality_score(raw_hist, filt, str(pd_date.date()))
            final = {}
            for s in filt:
                sc = combined.get(s, -999)
                if s not in bnc:
                    sc *= BP
                sc += (qual.get(s, 0.5) - 0.5) * QC
                final[s] = sc
            top5 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
            sids_top = [s for s, _ in top5]
            _, w = equal_weight_allocate(sids_top)
            ret = 0.0
            for sid, wgt in zip(sids_top, w):
                r1 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t1)]
                r5 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t5)]
                if len(r1) > 0 and len(r5) > 0:
                    ret += (float(r5.iloc[0]['开盘']) - float(r1.iloc[0]['开盘'])) / float(r1.iloc[0]['开盘']) * wgt
            results.append({'Week': label, 'Seed': seed, 'Strategy': slabel, 'Return': ret})
        print(f'  Seed={seed} done')

# ── Report ──
df = pd.DataFrame(results)
print()
print('=' * 70)
print('StatArb Ensemble Results (MC=20, 5-seed)')
print('=' * 70)
print(f'{"Strategy":<35s} {"W1":>8s} {"W2":>8s} {"W3":>8s} {"W4":>8s} {"Mean":>8s}')
print('-' * 70)

for s in df['Strategy'].unique():
    vals = []
    row = []
    for wk in ['W1', 'W2', 'W3', 'W4']:
        sub = df[(df.Strategy == s) & (df.Week == wk)]
        if len(sub) > 0:
            v = sub.Return.mean() * 100
            row.append(f'{v:>+7.2f}%')
            vals.append(v)
        else:
            row.append('    N/A')
    mean_v = np.mean(vals) if vals else 0
    print(f'{s:<35s} {" ".join(row)} {mean_v:>+7.2f}%')

print('\nDone!')
