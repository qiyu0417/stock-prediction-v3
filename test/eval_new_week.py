"""Evaluate new week (6/15-6/22) with best models. MC=20 + 5 seeds."""
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

# ── Data ──
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


def load_model(mdir, nf, gnn=False):
    if gnn:
        _em.StockTransformerExpert.__init__ = _gnn_trans_init
        _em.ConvStockExpert.__init__ = _gnn_conv_init
        StockTransformerExpert.__init__ = _gnn_trans_init
        ConvStockExpert.__init__ = _gnn_conv_init
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
        m = StockTransformerExpert(nf, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(
            nf, e, ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        ms.append(m)
    return ms


G = load_model('model/stock_emb_8_gnn_corr', len(fcols_all), gnn=True)
_em.StockTransformerExpert.__init__ = _orig_ti
_em.ConvStockExpert.__init__ = _orig_ci
StockTransformerExpert.__init__ = _orig_ti
ConvStockExpert.__init__ = _orig_ci
H = load_model('model/stock_emb_8_hybrid', len(fcols_all))
A = load_model('model/stock_emb_8_alpha158', len(alpha_f))


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
                if isinstance(o, tuple): o = o[0]
                ps.append(o[0].cpu().numpy())
        all_s.append(np.mean(ps, axis=0))
    return np.mean(all_s, axis=0)


# ═══════════════════════════════════════
# True W4: predict 6/18, T+1=6/22, T+5=6/26 (5 trading days)
# ═══════════════════════════════════════
pd_str = pd.to_datetime('2026-06-18')
t1 = pd.to_datetime('2026-06-22')
t5 = pd.to_datetime('2026-06-26')
ref_str = str(pd_str.date())
print(f'W4: predict={pd_str.date()}, T+1={t1.date()}, T+5={t5.date()}')

seqG, sG, vG = build_seq(pG, pd_str, fcols_all, len(fcols_all))
seqH, sH, vH = build_seq(pH, pd_str, fcols_all, len(fcols_all))
seqA, sA, vA = build_seq(pA, pd_str, alpha_f, len(alpha_f))

seeds = [42, 123, 456, 789, 1024]
print(f'MC={MC}, {len(seeds)} seeds...')

results = []
for seed in seeds:
    set_seed(seed)
    rG = mc_infer(G, torch.FloatTensor(seqG).to(device))
    rH = mc_infer(H, torch.FloatTensor(seqH).to(device))
    rA = mc_infer(A, torch.FloatTensor(seqA).to(device))
    raw_hist = processed_raw[processed_raw['日期'] <= pd_str]

    gM = {s: float(rG[i]) for i, s in enumerate(sG) if vG[i]}
    hM = {s: float(rH[i]) for i, s in enumerate(sH) if vH[i]}
    aM = {s: float(rA[i]) for i, s in enumerate(sA) if vA[i]}
    common = sorted(set(gM) & set(hM) & set(aM))

    strategies = [
        ('CorrG+H+A (0.1/0.4/0.5)', 0.1, 0.4, 0.5),
        ('H+A (0.45/0.55)', 0.0, 0.45, 0.55),
        ('Hybrid only', 0.0, 1.0, 0.0),
        ('CorrG only', 1.0, 0.0, 0.0),
    ]

    for label, wg, wh, wa in strategies:
        combined = {s: wg * gM.get(s, 0) + wh * hM.get(s, 0) + wa * aM.get(s, 0) for s in common}
        filt = volatility_filter(raw_hist, list(common), ref_str, top_pct=VP)
        if len(filt) < 5: continue
        bnc = bounce_confirm(raw_hist, filt, ref_str, threshold=BT)
        qual = compute_quality_score(raw_hist, filt, ref_str)
        final = {}
        for s in filt:
            sc = combined.get(s, -999)
            if s not in bnc: sc *= BP
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
        results.append({'Seed': seed, 'Strategy': label, 'Return': ret,
                        'Top5': ','.join(sids_top)})
    print(f'  Seed={seed} done')

df = pd.DataFrame(results)
print()
print('=' * 60)
print('W4 (6/22-6/26, 5d): MC=20 5-seed')
print('=' * 60)
for s in df['Strategy'].unique():
    sub = df[df['Strategy'] == s]
    print(f'  {s:35s}: {sub.Return.mean() * 100:+.2f}% ±{sub.Return.std() * 100:.2f}%')

print()
print('=' * 60)
print('Top-5 picks (seed=42):')
print('=' * 60)
for s in df['Strategy'].unique():
    t5 = df[(df.Strategy == s) & (df.Seed == 42)]['Top5'].values[0]
    print(f'  {s}: {t5}')
print('\nDone!')
