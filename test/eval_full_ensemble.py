"""Full ensemble evaluation: up to 6 models with grid search weights + regime-adaptive."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, joblib
from train import (feature_cloums_map, feature_engineer_func_map, _build_label_and_clean,
                   set_seed, _ALPHA_158_COLS)
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate
from config_stock_emb_8 import FEATURE_NUM
from market_regime import compute_market_regime

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

# ── Model loading infrastructure ──
import ensemble_models as _em
_orig_ti, _orig_ci = _em.StockTransformerExpert.__init__, _em.ConvStockExpert.__init__
from ensemble_models import (StockTransformerExpert, ConvStockExpert,
                             StatArbRegressionExpert, AdversarialStockExpert)


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


class BrownianWrapper(torch.nn.Module):
    """Wrapper for BrownianNoiseExpert to provide standard forward signature."""
    def __init__(self, input_dim, expert_config, num_stocks):
        super().__init__()
        from ensemble_models import BrownianNoiseExpert
        self.expert = BrownianNoiseExpert(input_dim, expert_config, num_stocks)

    def forward(self, src):
        return self.expert.forward(src, epoch_progress=0.0, add_noise=False)

    def predict_with_mc_dropout(self, src, num_samples=20):
        return self.expert.predict_with_mc_dropout(src, num_samples)


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


# ── Model loader registry ──
def load_standard_models(mdir, nf, ns, use_features=None):
    """Load standard (transformer/conv) ensemble."""
    if use_features is None:
        use_features = nf
    with open(os.path.join(mdir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    ms = []
    for ec in cfg['expert_configs']:
        p = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(p):
            continue
        e = dict(ec)
        e['stock_embed_dim'] = emb
        m = StockTransformerExpert(use_features, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(
            use_features, e, ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        ms.append(m)
    return ms


def load_gnn_models(mdir, nf, ns):
    _em.StockTransformerExpert.__init__ = _gnn_init(StockTransformerExpert)
    _em.ConvStockExpert.__init__ = _gnn_init(ConvStockExpert, 'hidden_channels')
    StockTransformerExpert.__init__ = _gnn_init(StockTransformerExpert)
    ConvStockExpert.__init__ = _gnn_init(ConvStockExpert, 'hidden_channels')
    with open(os.path.join(mdir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    emb = cfg.get('stock_embed_dim', 8)
    ms = []
    for ec in cfg['expert_configs']:
        p = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(p):
            continue
        e = dict(ec)
        e['stock_embed_dim'] = emb
        m = StockTransformerExpert(nf, e, ns) if ec['type'] == 'transformer' else ConvStockExpert(nf, e, ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        ms.append(m)
    _em.StockTransformerExpert.__init__ = _orig_ti
    _em.ConvStockExpert.__init__ = _orig_ci
    StockTransformerExpert.__init__ = _orig_ti
    ConvStockExpert.__init__ = _orig_ci
    return ms


def load_statarb_models(mdir, nf, ns):
    with open(os.path.join(mdir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    ms = []
    for ec in cfg['expert_configs']:
        p = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(p):
            continue
        m = StatArbRegressionExpert(nf, dict(ec), ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        ms.append(m)
    return ms


def load_brownian_models(mdir, nf, ns):
    with open(os.path.join(mdir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    ms = []
    for ec in cfg['expert_configs']:
        p = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(p):
            continue
        m = BrownianWrapper(nf, dict(ec), ns)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        ms.append(m)
    return ms


def load_adversarial_models(mdir, nf, ns):
    with open(os.path.join(mdir, 'ensemble_config.json')) as f:
        cfg = json.load(f)
    ms = []
    for ec in cfg['expert_configs']:
        p = os.path.join(mdir, f'expert_{ec["name"]}.pth')
        if not os.path.exists(p):
            continue
        base_cfg = dict(ec)
        if ec['base_type'] == 'transformer':
            base = StockTransformerExpert(nf, base_cfg, ns)
            d_model = ec['d_model']
        else:
            base = ConvStockExpert(nf, base_cfg, ns)
            d_model = ec['hidden_channels']
        num_domains = cfg.get('num_time_domains', 12)
        adv_lambda = cfg.get('adv_lambda', 0.1)
        m = AdversarialStockExpert(base, d_model, num_domains, adv_lambda)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.train()
        ms.append(m)
    return ms


def preproc(mdir):
    with open(os.path.join(mdir, 'winsor_bounds.json')) as f:
        wb = json.load(f)
    sc = joblib.load(os.path.join(mdir, 'scaler.pkl'))
    # Only copy needed columns to save memory
    needed = fcols_all + ['日期', '股票代码']
    p = processed_raw[needed].copy()
    p[fcols_all] = p[fcols_all].replace([np.inf, -np.inf], np.nan).dropna(subset=fcols_all)
    for col, (lo, hi) in wb.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    p[fcols_all] = sc.transform(p[fcols_all])
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
        return None, None
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


def preproc_alpha(mdir):
    """Preprocess using Alpha158 features only."""
    with open(os.path.join(mdir, 'winsor_bounds.json')) as f:
        wb = json.load(f)
    sc = joblib.load(os.path.join(mdir, 'scaler.pkl'))
    needed = alpha_f + ['日期', '股票代码']
    p = processed_raw[needed].copy()
    for col, (lo, hi) in wb.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    p[alpha_f] = sc.transform(p[alpha_f])
    return p


# ── Preprocess ──
pG = preproc('model/stock_emb_8_gnn_corr')
pH = preproc('model/stock_emb_8_hybrid')
pA = preproc_alpha('model/stock_emb_8_alpha158')
pS, pB, pAdv = None, None, None

n_feats = len(fcols_all)
n_alpha = len(alpha_f)
with open('model/stock_emb_8_hybrid/ensemble_config.json') as f:
    num_stocks = json.load(f)['num_stocks']

# ── Load models (skip if not trained yet) ──
print('\nLoading models...')
G = load_gnn_models('model/stock_emb_8_gnn_corr', n_feats, num_stocks)
print(f'  CorrGNN: {len(G)} experts')

H = load_standard_models('model/stock_emb_8_hybrid', n_feats, num_stocks)
print(f'  Hybrid: {len(H)} experts')

A = load_standard_models('model/stock_emb_8_alpha158', n_alpha, num_stocks)
print(f'  Alpha158: {len(A)} experts')

has_statarb = os.path.exists('model/stock_emb_8_statarb/ensemble_config.json')
S = []
if has_statarb:
    pS = preproc('model/stock_emb_8_statarb')
    S = load_statarb_models('model/stock_emb_8_statarb', n_feats, num_stocks)
    print(f'  StatArb: {len(S)} experts')
else:
    print('  StatArb: not trained yet')

has_brownian = os.path.exists('model/stock_emb_8_brownian/ensemble_config.json')
B = []
if has_brownian:
    pB = preproc('model/stock_emb_8_brownian')
    B = load_brownian_models('model/stock_emb_8_brownian', n_feats, num_stocks)
    print(f'  BrownianNoise: {len(B)} experts')
else:
    print('  BrownianNoise: not trained yet')

has_adversarial = os.path.exists('model/stock_emb_8_adversarial/ensemble_config.json')
Adv = []
if has_adversarial:
    pAdv = preproc('model/stock_emb_8_adversarial')
    Adv = load_adversarial_models('model/stock_emb_8_adversarial', n_feats, num_stocks)
    print(f'  Adversarial: {len(Adv)} experts')
else:
    print('  Adversarial: not trained yet')


# ── Compute regime features ──
def get_regime_weights(predict_date):
    """Simple heuristic regime-adaptive weights based on market state."""
    regime = compute_market_regime(processed_raw, fcols_all, all_sids, predict_date)

    # Momentum strength: combine trend and recent returns
    ret_5d = regime.get('ret_5d', 0)
    ret_20d = regime.get('ret_20d', 0)
    momentum_score = (ret_5d + ret_20d) / 0.06 if abs(ret_5d + ret_20d) > 0 else 0
    momentum_score = np.clip((momentum_score + 1) / 2, 0, 1)  # normalize to 0-1

    # Breadth: how many stocks above MA20
    breadth = regime.get('breadth_score', 0.5)  # 0=good, 1=bad
    breadth_ok = 1 - breadth  # invert: 1=good

    # Volatility regime
    vol_score = regime.get('volatility_score', 0.5)  # 0=calm, 1=high

    # Adaptive weights
    # High momentum -> more Alpha158 (momentum factors)
    # High vol -> more GNN (graph structure helps in turbulence)
    # Low breadth -> more Hybrid (dense attention is safer)
    # StatArb (mean reversion) -> more weight when momentum is low

    w_alpha158 = 0.1 + 0.3 * momentum_score  # 0.1-0.4
    w_gnn = 0.0 + 0.2 * vol_score  # 0.0-0.2
    w_hybrid = 0.3 + 0.3 * breadth_ok  # 0.3-0.6
    w_statarb = 0.1 + 0.15 * (1 - momentum_score)  # 0.1-0.25 (more when momentum weak)

    # Normalize to sum=1
    total = w_alpha158 + w_gnn + w_hybrid + w_statarb
    if total > 0:
        w_alpha158 /= total
        w_gnn /= total
        w_hybrid /= total
        w_statarb /= total

    return {
        'w_alpha158': w_alpha158, 'w_gnn': w_gnn,
        'w_hybrid': w_hybrid, 'w_statarb': w_statarb,
        'momentum_score': momentum_score, 'vol_score': vol_score,
        'breadth_ok': breadth_ok,
    }


# ── Evaluation ──
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

    seqG, sG, vG = build_seq(pG, pd_date, fcols_all, n_feats)
    seqH, sH, vH = build_seq(pH, pd_date, fcols_all, n_feats)
    seqA, sA, vA = build_seq(pA, pd_date, alpha_f, n_alpha)
    seqS, sS, vS = (None, None, None) if not S else build_seq(pS, pd_date, fcols_all, n_feats)
    seqB, sB, vB = (None, None, None) if not B else build_seq(pB, pd_date, fcols_all, n_feats)
    seqAdv, sAdv, vAdv = (None, None, None) if not Adv else build_seq(pAdv, pd_date, fcols_all, n_feats)

    regime_w = get_regime_weights(pd_date)
    print(f'  Momentum={regime_w["momentum_score"]:.2f} Vol={regime_w["vol_score"]:.2f} '
          f'BreadthOK={regime_w["breadth_ok"]:.2f}')
    print(f'  Adaptive w: G={regime_w["w_gnn"]:.2f} H={regime_w["w_hybrid"]:.2f} '
          f'A={regime_w["w_alpha158"]:.2f} S={regime_w["w_statarb"]:.2f}')

    for seed in seeds:
        set_seed(seed)
        stG = torch.FloatTensor(seqG).to(device)
        stH = torch.FloatTensor(seqH).to(device)
        stA = torch.FloatTensor(seqA).to(device)

        rG = mc_infer(G, stG)
        rH = mc_infer(H, stH)
        rA = mc_infer(A, stA)
        rS = mc_infer(S, torch.FloatTensor(seqS).to(device)) if S else None
        rB_out = mc_infer(B, torch.FloatTensor(seqB).to(device)) if B else None
        rAdv_out = mc_infer(Adv, torch.FloatTensor(seqAdv).to(device)) if Adv else None

        raw_hist = processed_raw[processed_raw['日期'] <= pd_date]

        # Build per-model predictions
        models = {
            'GNN': ({s: float(rG[i]) for i, s in enumerate(sG) if vG[i]} if rG is not None else {}),
            'Hybrid': {s: float(rH[i]) for i, s in enumerate(sH) if vH[i]},
            'Alpha158': {s: float(rA[i]) for i, s in enumerate(sA) if vA[i]},
        }
        if rS is not None:
            models['StatArb'] = {s: float(rS[i]) for i, s in enumerate(sS) if vS[i]}
        if rB_out is not None:
            models['Brownian'] = {s: float(rB_out[i]) for i, s in enumerate(sB) if vB[i]}
        if rAdv_out is not None:
            models['Adversarial'] = {s: float(rAdv_out[i]) for i, s in enumerate(sAdv) if vAdv[i]}

        # Common stock set
        common = sorted(set.intersection(*[set(m.keys()) for m in models.values()]))

        # ── Strategy definitions ──
        strategies = []

        # Baseline: 3-model (current best)
        if all(k in models for k in ['GNN', 'Hybrid', 'Alpha158']):
            strategies.append(('CorrG+H+A (0.1/0.4/0.5)', {
                'GNN': 0.1, 'Hybrid': 0.4, 'Alpha158': 0.5}))

        # Hybrid only
        strategies.append(('Hybrid only', {'Hybrid': 1.0}))

        # Add StatArb blends if available
        if 'StatArb' in models:
            strategies.append(('H+A+S (0.35/0.4/0.25)', {
                'Hybrid': 0.35, 'Alpha158': 0.4, 'StatArb': 0.25}))
            strategies.append(('G+H+A+S (0.1/0.35/0.4/0.15)', {
                'GNN': 0.1, 'Hybrid': 0.35, 'Alpha158': 0.4, 'StatArb': 0.15}))

        # Add Brownian if available
        if 'Brownian' in models:
            strategies.append(('H+A+B (0.35/0.4/0.25)', {
                'Hybrid': 0.35, 'Alpha158': 0.4, 'Brownian': 0.25}))

        # Add Adversarial if available
        if 'Adversarial' in models:
            strategies.append(('H+A+Adv (0.35/0.4/0.25)', {
                'Hybrid': 0.35, 'Alpha158': 0.4, 'Adversarial': 0.25}))

        # Full ensemble (all available)
        active_models = sorted(models.keys())
        if len(active_models) >= 5:
            # Equal weight for full ensemble
            ew = {m: 1.0 / len(active_models) for m in active_models}
            strategies.append((f'All-{len(active_models)} equal', ew))

        # Regime-adaptive
        if 'StatArb' in models:
            strategies.append(('Regime-Adaptive', {
                'GNN': regime_w['w_gnn'], 'Hybrid': regime_w['w_hybrid'],
                'Alpha158': regime_w['w_alpha158'], 'StatArb': regime_w['w_statarb']}))
        else:
            total_3 = regime_w['w_gnn'] + regime_w['w_hybrid'] + regime_w['w_alpha158']
            if total_3 > 0:
                strategies.append(('Regime-Adaptive', {
                    'GNN': regime_w['w_gnn'] / total_3,
                    'Hybrid': regime_w['w_hybrid'] / total_3,
                    'Alpha158': regime_w['w_alpha158'] / total_3}))

        for slabel, weights in strategies:
            combined = {s: sum(weights.get(m, 0) * models[m].get(s, 0) for m in weights)
                       for s in common}
            filt = volatility_filter(raw_hist, list(common), str(pd_date.date()), top_pct=VP)
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
print('Full Ensemble Results (MC=20, 5-seed)')
print('=' * 70)
print(f'{"Strategy":<35s} {"W1":>8s} {"W2":>8s} {"W3":>8s} {"W4":>8s} {"Mean":>8s} {"Std":>8s}')
print('-' * 70)

for s in sorted(df['Strategy'].unique()):
    vals = []
    row = []
    for wk in ['W1', 'W2', 'W3', 'W4']:
        sub = df[(df.Strategy == s) & (df.Week == wk)]
        if len(sub) > 0:
            v = sub.Return.mean() * 100
            row.append(f'{v:>+7.2f}%')
            vals.append(v)
        else:
            row.append(f'{">8s"}')
    mean_v = np.mean(vals) if vals else 0
    std_v = np.std(vals) if vals else 0
    print(f'{s:<35s} {" ".join(row)} {mean_v:>+7.2f}% {std_v:>7.2f}%')

print('\nDone!')
