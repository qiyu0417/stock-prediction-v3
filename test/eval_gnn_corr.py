"""Evaluate GNN model: MC=20 + 5 seeds on June test set."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np
import pandas as pd
import torch
import joblib
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC = 20
SEQ = 60
device = torch.device('cuda')

# ── Load data ──
print("Loading data...")
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

from config_stock_emb_8 import FEATURE_NUM
fe = feature_engineer_func_map[FEATURE_NUM]
fcols_all = feature_cloums_map[FEATURE_NUM]
groups = [g.reset_index(drop=True) for _, g in full_df.groupby('股票代码', sort=False) if len(g) >= SEQ + 10]
processed_raw = pd.concat([fe(g) for g in groups]).reset_index(drop=True)
processed_raw['instrument'] = processed_raw['股票代码'].map(sid2idx)
processed_raw = processed_raw.dropna(subset=['instrument']).copy()
processed_raw['instrument'] = processed_raw['instrument'].astype(np.int64)
processed_raw = _build_label_and_clean(processed_raw, drop_small_open=True)
alpha_f = [f for f in _ALPHA_158_COLS if f in fcols_all]

# ── Load models (with GNN monkey-patch) ──
import ensemble_models as _em
_orig_ti = _em.StockTransformerExpert.__init__
_orig_ci = _em.ConvStockExpert.__init__
from ensemble_models import StockTransformerExpert, ConvStockExpert

# GNN module
class GraphStockConv(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def set_adjacency(self, adj_norm):
        self.register_buffer('adj_norm', adj_norm)

    def forward(self, stock_features):
        B, N, D = stock_features.shape
        adj = self.adj_norm[:N, :N].to(stock_features.device)
        support = self.linear(stock_features)
        out = torch.bmm(adj.unsqueeze(0).expand(B, -1, -1), support)
        out = torch.nn.functional.relu(out)
        return self.norm(stock_features + self.dropout(out))


def build_industry_adjacency(stock_ids):
    ind_df = pd.read_csv('data/industry.csv', dtype={'股票代码': str})
    ind_df['股票代码'] = ind_df['股票代码'].astype(str).str.zfill(6)
    sector_map = {}
    for _, row in ind_df.iterrows():
        sector_map[row['股票代码']] = row.get('sector', 'Z')
    N = len(stock_ids)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        si = sector_map.get(stock_ids[i], 'Z')
        for j in range(N):
            if si == sector_map.get(stock_ids[j], 'Z'):
                adj[i, j] = 1.0
    adj = adj + np.eye(N, dtype=np.float32)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    return torch.FloatTensor(adj_norm)


_gnn_adj = build_industry_adjacency(all_sids)


def _gnn_trans_init(self, input_dim, expert_config, num_stocks):
    _orig_ti(self, input_dim, expert_config, num_stocks)
    gnn = GraphStockConv(self.d_model, expert_config.get('dropout', 0.1))
    gnn.set_adjacency(_gnn_adj)
    self.cross_stock_attention = gnn


def _gnn_conv_init(self, input_dim, expert_config, num_stocks):
    _orig_ci(self, input_dim, expert_config, num_stocks)
    d = expert_config.get('hidden_channels', 256)
    gnn = GraphStockConv(d, expert_config.get('dropout', 0.1))
    gnn.set_adjacency(_gnn_adj)
    self.cross_stock_attention = gnn


_em.StockTransformerExpert.__init__ = _gnn_trans_init
_em.ConvStockExpert.__init__ = _gnn_conv_init
StockTransformerExpert.__init__ = _gnn_trans_init
ConvStockExpert.__init__ = _gnn_conv_init


def load_model(mdir, nf):
    with open(os.path.join(mdir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    with open('model/stock_emb_8_hybrid/ensemble_config.json') as f2:
        ns = json.load(f2)['num_stocks']
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        e = dict(ec)
        e['stock_embed_dim'] = emb
        m = StockTransformerExpert(nf, e, ns) if ec['type'] == 'transformer' \
            else ConvStockExpert(nf, e, ns)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        models.append(m)
    return models


def preprocess(mdir):
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


print("Loading models...")

# Load GNN with monkey-patch
G = load_model('model/stock_emb_8_gnn_corr', len(fcols_all))
p_g = preprocess('model/stock_emb_8_gnn_corr')

# Restore original inits
_em.StockTransformerExpert.__init__ = _orig_ti
_em.ConvStockExpert.__init__ = _orig_ci
StockTransformerExpert.__init__ = _orig_ti
ConvStockExpert.__init__ = _orig_ci

# Load Hybrid and Alpha158 with original inits
H = load_model('model/stock_emb_8_hybrid', len(fcols_all))
A = load_model('model/stock_emb_8_alpha158', len(alpha_f))
p_h = preprocess('model/stock_emb_8_hybrid')
p_a = preprocess('model/stock_emb_8_alpha158')
print(f"  GNN={len(G)} experts, Hybrid={len(H)} experts, Alpha158={len(A)} experts")


# ── Inference helpers ──
def build_seq(p, ref_date, fcols, nf):
    hist = p[p['日期'] <= ref_date]
    sids = sorted(hist['股票代码'].unique())
    seq = np.zeros((1, len(sids), SEQ, nf), dtype=np.float32)
    valid = np.zeros(len(sids), dtype=bool)
    for i, sid in enumerate(sids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQ:
            seq[0, i] = sd[fcols].values[-SEQ:].astype(np.float32)
            valid[i] = True
    return seq, sids, valid


def mc_infer(models, seq_t, n_samples=MC):
    all_s = []
    for _ in range(n_samples):
        ps = []
        for m in models:
            with torch.no_grad():
                out = m(seq_t)
                if isinstance(out, tuple):
                    out = out[0]
                ps.append(out[0].cpu().numpy())
        all_s.append(np.mean(ps, axis=0))
    return np.mean(all_s, axis=0)


def compute_top5_return(top5_sids, t1_date, t5_date):
    sids, _ = equal_weight_allocate(top5_sids)
    ret = 0.0
    for sid, wgt in zip(sids, [0.2] * 5):
        r1 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t1_date)]
        r5 = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t5_date)]
        if len(r1) > 0 and len(r5) > 0:
            sr = (float(r5.iloc[0]['开盘']) - float(r1.iloc[0]['开盘'])) / float(r1.iloc[0]['开盘'])
            ret += sr * wgt
    return ret


# ═══════════════════════════════════════════════════════════
# MC=20 5-seed evaluation
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MC=20 5-seed evaluation: GNN + blends")
print(f"{'='*60}")

weeks = [
    ('W1', pd.to_datetime('2026-05-29'), pd.to_datetime('2026-06-01'),
     pd.to_datetime('2026-06-05'), 5),
    ('W2', pd.to_datetime('2026-06-05'), pd.to_datetime('2026-06-08'),
     pd.to_datetime('2026-06-12'), 5),
    ('W3', pd.to_datetime('2026-06-12'), pd.to_datetime('2026-06-15'),
     pd.to_datetime('2026-06-18'), 4),
]

seeds = [42, 123, 456, 789, 1024]

strategies = [
    # Baselines
    ('GNN only', 1.0, 0.0, 0.0),
    ('Hybrid only', 0.0, 1.0, 0.0),
    ('Alpha158 only', 0.0, 0.0, 1.0),
    # 2-model blends
    ('H+A w=0.45/0.55', 0.0, 0.45, 0.55),
    ('G+H w=0.3/0.7', 0.3, 0.7, 0.0),
    # 3-model blends
    ('G=0.1 H=0.4 A=0.5', 0.1, 0.4, 0.5),
    ('G=0.1 H=0.5 A=0.4', 0.1, 0.5, 0.4),
    ('G=0.15 H=0.45 A=0.4', 0.15, 0.45, 0.4),
    ('G=0.2 H=0.4 A=0.4', 0.2, 0.4, 0.4),
    ('G=0.2 H=0.5 A=0.3', 0.2, 0.5, 0.3),
    ('G=0.3 H=0.4 A=0.3', 0.3, 0.4, 0.3),
]

all_results = []

for seed in seeds:
    set_seed(seed)
    print(f"\n--- Seed={seed} ---")

    for wname, pd_str, t1_date, t5_date, ndays in weeks:
        ref_date = pd_str
        ref_str = str(pd_str.date())

        seq_g, sids_g, valid_g = build_seq(p_g, ref_date, fcols_all, len(fcols_all))
        seq_h, sids_h, valid_h = build_seq(p_h, ref_date, fcols_all, len(fcols_all))
        seq_a, sids_a, valid_a = build_seq(p_a, ref_date, alpha_f, len(alpha_f))

        raw_g = mc_infer(G, torch.FloatTensor(seq_g).to(device))
        raw_h = mc_infer(H, torch.FloatTensor(seq_h).to(device))
        raw_a = mc_infer(A, torch.FloatTensor(seq_a).to(device))

        raw_hist = processed_raw[processed_raw['日期'] <= ref_date]

        for strat_name, wg, wh, wa in strategies:
            g_map = {s: float(raw_g[i]) for i, s in enumerate(sids_g) if valid_g[i]}
            h_map = {s: float(raw_h[i]) for i, s in enumerate(sids_h) if valid_h[i]}
            a_map = {s: float(raw_a[i]) for i, s in enumerate(sids_a) if valid_a[i]}
            common = sorted(set(g_map.keys()) & set(h_map.keys()) & set(a_map.keys()))
            if len(common) < 10:
                continue

            combined = {sid: wg * g_map.get(sid, 0) + wh * h_map.get(sid, 0) +
                         wa * a_map.get(sid, 0) for sid in common}

            filt = volatility_filter(raw_hist, list(common), ref_str, top_pct=0.95)
            if len(filt) < 5:
                continue
            bnc = bounce_confirm(raw_hist, filt, ref_str, threshold=0.008)
            qual = compute_quality_score(raw_hist, filt, ref_str)

            final = {}
            for sid in filt:
                s = combined.get(sid, -999)
                if sid not in bnc:
                    s *= 0.92
                s += (qual.get(sid, 0.5) - 0.5) * 0.05
                final[sid] = s

            top5 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:5]
            ret = compute_top5_return([s for s, _ in top5], t1_date, t5_date)

            all_results.append({
                'Seed': seed, 'Week': wname, 'Strategy': strat_name,
                'Return': ret, 'Top5': ','.join([s for s, _ in top5]),
            })

        print(f"  {wname} ok")

# ── Summary ──
df = pd.DataFrame(all_results)
print(f"\n{'='*70}")
print("RESULTS: MC=20 5-seed mean returns")
print(f"{'='*70}")

for strat_name, _, _, _ in strategies:
    sub = df[df['Strategy'] == strat_name]
    if len(sub) > 0:
        print(f"  {strat_name:20s}: {sub['Return'].mean()*100:+.2f}% ±{sub['Return'].std()*100:.2f}%")

print(f"\n{'='*70}")
print("WEEKLY BREAKDOWN")
print(f"{'='*70}")
for week_name, _, _, _, _ in weeks:
    print(f"\n  {week_name}:")
    sub_w = df[df['Week'] == week_name]
    for strat_name, _, _, _ in strategies:
        sub = sub_w[sub_w['Strategy'] == strat_name]
        if len(sub) > 0:
            print(f"    {strat_name:20s}: {sub['Return'].mean()*100:+.2f}% ±{sub['Return'].std()*100:.2f}%")

print(f"\nDone!")
