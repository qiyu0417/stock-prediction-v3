"""Evaluate GNN with rolling 60-day correlation graph — test if static graph is stale."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, joblib
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate
from config_stock_emb_8 import FEATURE_NUM

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

# ── Models ──
import ensemble_models as _em
_orig_ti, _orig_ci = _em.StockTransformerExpert.__init__, _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert


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


def build_adj(stock_ids, ref_date, lookback_days=60):
    """Build correlation adjacency using only data within [ref_date - lookback_days, ref_date]."""
    cutoff = ref_date - pd.Timedelta(days=lookback_days)
    window = full_df[(full_df['日期'] >= cutoff) & (full_df['日期'] <= ref_date)]
    pvt = window.pivot(index='日期', columns='股票代码', values='涨跌幅')
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


_gnn_adj = None


def _gnn_trans_init(s, i, c, n):
    _orig_ti(s, i, c, n)
    g = GraphStockConv(s.d_model, c.get('dropout', 0.1))
    g.set_adjacency(_gnn_adj)
    s.cross_stock_attention = g


def _gnn_conv_init(s, i, c, n):
    _orig_ci(s, i, c, n)
    d = c.get('hidden_channels', 256)
    g = GraphStockConv(d, c.get('dropout', 0.1))
    g.set_adjacency(_gnn_adj)
    s.cross_stock_attention = g


def load_gnn_models(mdir, nf):
    _em.StockTransformerExpert.__init__ = _gnn_trans_init
    _em.ConvStockExpert.__init__ = _gnn_conv_init
    StockTransformerExpert.__init__ = _gnn_trans_init
    ConvStockExpert.__init__ = _gnn_conv_init
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
        m = StockTransformerExpert(nf, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(
            nf, e, ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        ms.append(m)
    return ms


def restore_inits():
    _em.StockTransformerExpert.__init__ = _orig_ti
    _em.ConvStockExpert.__init__ = _orig_ci
    StockTransformerExpert.__init__ = _orig_ti
    ConvStockExpert.__init__ = _orig_ci


def preproc(mdir):
    with open(os.path.join(mdir, 'winsor_bounds.json')) as f:
        wb = json.load(f)
    sc = joblib.load(os.path.join(mdir, 'scaler.pkl'))
    p = processed_raw.copy()
    p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
    for col, (lo, hi) in wb.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    p[fcols_all] = sc.transform(p[fcols_all])
    return p


pG = preproc('model/stock_emb_8_gnn_corr')
pH = preproc('model/stock_emb_8_hybrid')
pA = preproc('model/stock_emb_8_alpha158')


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


def eval_week(predict_date, t1_date, t5_date, label, gnn_mdir):
    """Evaluate one week with dynamic GNN graph."""
    global _gnn_adj
    print(f'\n{"=" * 60}')
    print(f'{label}: predict={predict_date.date()}, T+1={t1_date.date()}, T+5={t5_date.date()}')
    print(f'{"=" * 60}')

    # Build dynamic adjacency from rolling 60-day window
    _gnn_adj = build_adj(all_sids, predict_date, lookback_days=60)
    deg = (_gnn_adj.numpy() > 0).sum(1).mean()
    print(f'  Dynamic corr graph: avg_deg={deg:.1f}')

    G = load_gnn_models(gnn_mdir, len(fcols_all))
    restore_inits()

    # Load standard models
    _em.StockTransformerExpert.__init__ = _orig_ti
    _em.ConvStockExpert.__init__ = _orig_ci
    StockTransformerExpert.__init__ = _orig_ti
    ConvStockExpert.__init__ = _orig_ci

    with open('model/stock_emb_8_hybrid/ensemble_config.json') as f:
        hcfg = json.load(f)
    H = []
    for ec in hcfg['expert_configs']:
        p = os.path.join('model/stock_emb_8_hybrid', f'expert_{ec["name"]}.pth')
        if not os.path.exists(p):
            continue
        e = dict(ec)
        e['stock_embed_dim'] = 8
        nf = len(fcols_all)
        with open('model/stock_emb_8_hybrid/ensemble_config.json') as f2:
            ns = json.load(f2)['num_stocks']
        m = StockTransformerExpert(nf, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(nf, e, ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        H.append(m)

    with open('model/stock_emb_8_alpha158/ensemble_config.json') as f:
        acfg = json.load(f)
    A = []
    for ec in acfg['expert_configs']:
        p = os.path.join('model/stock_emb_8_alpha158', f'expert_{ec["name"]}.pth')
        if not os.path.exists(p):
            continue
        e = dict(ec)
        e['stock_embed_dim'] = 8
        nf = len(alpha_f)
        with open('model/stock_emb_8_alpha158/ensemble_config.json') as f2:
            ns = json.load(f2)['num_stocks']
        m = StockTransformerExpert(nf, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(nf, e, ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        A.append(m)

    seqG, sG, vG = build_seq(pG, predict_date, fcols_all, len(fcols_all))
    seqH, sH, vH = build_seq(pH, predict_date, fcols_all, len(fcols_all))
    seqA, sA, vA = build_seq(pA, predict_date, alpha_f, len(alpha_f))

    seeds = [42, 123, 456, 789, 1024]
    results = []
    for seed in seeds:
        set_seed(seed)
        rG = mc_infer(G, torch.FloatTensor(seqG).to(device))
        rH = mc_infer(H, torch.FloatTensor(seqH).to(device))
        rA = mc_infer(A, torch.FloatTensor(seqA).to(device))
        raw_hist = processed_raw[processed_raw['日期'] <= predict_date]

        gM = {s: float(rG[i]) for i, s in enumerate(sG) if vG[i]}
        hM = {s: float(rH[i]) for i, s in enumerate(sH) if vH[i]}
        aM = {s: float(rA[i]) for i, s in enumerate(sA) if vA[i]}
        common = sorted(set(gM) & set(hM) & set(aM))

        strategies = [
            ('DynamicG+H+A (0.1/0.4/0.5)', 0.1, 0.4, 0.5),
            ('Hybrid only', 0.0, 1.0, 0.0),
            ('DynamicG only', 1.0, 0.0, 0.0),
        ]

        for slabel, wg, wh, wa in strategies:
            combined = {s: wg * gM.get(s, 0) + wh * hM.get(s, 0) + wa * aM.get(s, 0) for s in common}
            filt = volatility_filter(raw_hist, list(common), str(predict_date.date()), top_pct=VP)
            if len(filt) < 5:
                continue
            bnc = bounce_confirm(raw_hist, filt, str(predict_date.date()), threshold=BT)
            qual = compute_quality_score(raw_hist, filt, str(predict_date.date()))
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
                r1 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t1_date)]
                r5 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t5_date)]
                if len(r1) > 0 and len(r5) > 0:
                    ret += (float(r5.iloc[0]['开盘']) - float(r1.iloc[0]['开盘'])) / float(r1.iloc[0]['开盘']) * wgt
            results.append({'Week': label, 'Seed': seed, 'Strategy': slabel, 'Return': ret})
        print(f'  Seed={seed} done')

    # Clean up
    del G, H, A
    torch.cuda.empty_cache()
    return results


# ── Evaluate all 4 weeks ──
weeks = [
    (pd.to_datetime('2026-05-28'), pd.to_datetime('2026-06-01'), pd.to_datetime('2026-06-05'), 'W1'),
    (pd.to_datetime('2026-06-04'), pd.to_datetime('2026-06-08'), pd.to_datetime('2026-06-12'), 'W2'),
    (pd.to_datetime('2026-06-11'), pd.to_datetime('2026-06-15'), pd.to_datetime('2026-06-18'), 'W3'),
    (pd.to_datetime('2026-06-18'), pd.to_datetime('2026-06-22'), pd.to_datetime('2026-06-26'), 'W4'),
]

all_results = []
for pd_date, t1, t5, label in weeks:
    all_results.extend(eval_week(pd_date, t1, t5, label, 'model/stock_emb_8_gnn_corr'))

df = pd.DataFrame(all_results)

# ── Also load static GNN baseline results for comparison ──
# Static graph results from previous eval:
static_baseline = {
    'W1': {'CorrG+H+A': 3.39, 'Hybrid only': 0.92, 'CorrG only': 3.39},
    'W2': {'CorrG+H+A': 15.15, 'Hybrid only': 14.13, 'CorrG only': 11.02},
    'W3': {'CorrG+H+A': 17.90, 'Hybrid only': 15.84, 'CorrG only': 17.46},
    'W4': {'CorrG+H+A': 4.45, 'Hybrid only': 6.02, 'CorrG only': 2.87},
}

print()
print('=' * 70)
print('Dynamic (rolling 60d) vs Static (full history) Correlation Graph GNN')
print('=' * 70)
print(f'{"Week":<6} {"Strategy":<30} {"Static":>8} {"Dynamic":>8} {"Delta":>8}')
print('-' * 70)

for week in ['W1', 'W2', 'W3', 'W4']:
    for strat in ['DynamicG+H+A (0.1/0.4/0.5)', 'Hybrid only', 'DynamicG only']:
        dyn = df[(df.Week == week) & (df.Strategy == strat)]
        if len(dyn) == 0:
            continue
        dyn_avg = dyn.Return.mean() * 100
        static_key = strat.replace('DynamicG', 'CorrG')
        if static_key in static_baseline[week]:
            static_val = static_baseline[week][static_key]
            delta = dyn_avg - static_val
            print(f'{week:<6} {strat:<30} {static_val:>+7.2f}% {dyn_avg:>+7.2f}% {delta:>+7.2f}%')

print()
print('=' * 70)
print('Weekly mean comparison:')
print('=' * 70)
for strat in df['Strategy'].unique():
    sub = df[df['Strategy'] == strat]
    print(f'  {strat:<35s}: {sub.Return.mean() * 100:+.2f}%')

print('\nDone!')
