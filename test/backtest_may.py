"""Backtest EMA+Hybrid on May 2026 weekly"""
import sys, os, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from train import feature_cloums_map, feature_engineer_func_map, _build_label_and_clean, set_seed
from quality_filter import volatility_filter, bounce_confirm, compute_quality_score, equal_weight_allocate

MC_PASSES = 5
SEQUENCE_LENGTH = 60
device = torch.device('cuda')
set_seed(42)

# Load ALL data (train + test)
train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
train_df['股票代码'] = train_df['股票代码'].astype(str).str.zfill(6)
train_df['日期'] = pd.to_datetime(train_df['日期'], format='mixed')

test_df = pd.read_csv('data/test.csv', dtype={'股票代码': str})
test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
test_df['日期'] = pd.to_datetime(test_df['日期'], format='mixed')

full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.drop_duplicates(subset=['股票代码', '日期'], keep='last')
print(f"Full data: {full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}")

raw_data = full_df.copy()
all_stock_ids = sorted(full_df['股票代码'].unique())
stockid2idx = {s: i for i, s in enumerate(all_stock_ids)}

from config_stock_emb_8 import FEATURE_NUM
feature_engineer = feature_engineer_func_map[FEATURE_NUM]
feature_columns = feature_cloums_map[FEATURE_NUM]

# Preprocess
model_dir = 'model/stock_emb_8_hybrid'
with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
    cfg = json.load(f)
feature_dim = cfg['feature_dim']
num_stocks = cfg['num_stocks']
embed_dim = cfg.get('stock_embed_dim', 8)

scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
with open(os.path.join(model_dir, 'winsor_bounds.json'), 'r') as f:
    winsor_bounds = json.load(f)

df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
groups = [g.reset_index(drop=True) for _, g in df.groupby('股票代码', sort=False) if len(g) >= SEQUENCE_LENGTH + 10]
processed = pd.concat([feature_engineer(g) for g in groups]).reset_index(drop=True)
processed['instrument'] = processed['股票代码'].map(stockid2idx)
processed = processed.dropna(subset=['instrument']).copy()
processed['instrument'] = processed['instrument'].astype(np.int64)
processed = _build_label_and_clean(processed, drop_small_open=True)
processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
processed = processed.dropna(subset=feature_columns)
for col, (lo, hi) in winsor_bounds.items():
    if col in processed.columns:
        processed[col] = processed[col].clip(lo, hi)
processed[feature_columns] = scaler.transform(processed[feature_columns])
n_feats = len(feature_columns)

# Load both models
def load_experts(model_dir_local):
    import ensemble_models as _em
    _orig_fa = _em.FeatureAttention
    if 'rankglu' in model_dir_local.lower():
        class GFA(nn.Module):
            def __init__(self, d_model, dropout=0.1):
                super().__init__()
                self.attention = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1), nn.Softmax(dim=1))
                self.gate = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, d_model), nn.Sigmoid())
                self.dropout = nn.Dropout(dropout)
            def forward(self, x):
                a = self.attention(x)
                return self.dropout(torch.sum(x * a, dim=1) * self.gate(torch.sum(x * a, dim=1)))
        _em.FeatureAttention = GFA

    from ensemble_models import StockTransformerExpert, ConvStockExpert
    with open(os.path.join(model_dir_local, 'ensemble_config.json'), 'r') as f:
        cfg_local = json.load(f)
    embed_dim_local = cfg_local.get('stock_embed_dim', 8)
    expert_configs = cfg_local['expert_configs']
    models = []
    for ec in expert_configs:
        path = os.path.join(model_dir_local, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        ec_copy = dict(ec)
        ec_copy['stock_embed_dim'] = embed_dim_local
        if ec['type'] == 'transformer':
            model = StockTransformerExpert(feature_dim, ec_copy, num_stocks)
        else:
            model = ConvStockExpert(feature_dim, ec_copy, num_stocks)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.train()
        models.append(model)
    _em.FeatureAttention = _orig_fa
    return models

print('Loading EMA...')
ema_models = load_experts('model/stock_emb_8_ema')
print(f'  {len(ema_models)} experts')
print('Loading Hybrid...')
hybrid_models = load_experts('model/stock_emb_8_hybrid')
print(f'  {len(hybrid_models)} experts')

# May 2026 prediction Mondays
# Use Mondays in May, but aligned with available data (Chinese market calendar)
may_mondays = ['2026-05-04', '2026-05-11', '2026-05-18', '2026-05-25']

all_trade_dates = sorted(full_df['日期'].unique())
print(f'All dates range: {all_trade_dates[0]} to {all_trade_dates[-1]}')

results = []
for pred_date_str in may_mondays:
    pred_dt = pd.to_datetime(pred_date_str)
    # Find nearest available date <= pred_date
    valid_pred_dates = [d for d in all_trade_dates if d <= pred_dt]
    if not valid_pred_dates:
        print(f'\n{pred_date_str}: no available trading date before')
        continue
    actual_pred_date = valid_pred_dates[-1]

    hist = processed[processed['日期'] <= actual_pred_date]
    avail_stocks = hist['股票代码'].unique()
    stock_ids = sorted(avail_stocks)
    n_stocks = len(stock_ids)
    if n_stocks < 5:
        continue

    sequences = np.zeros((1, n_stocks, SEQUENCE_LENGTH, n_feats), dtype=np.float32)
    valid_mask = np.zeros(n_stocks, dtype=bool)
    for i, sid in enumerate(stock_ids):
        sd = hist[hist['股票代码'] == sid].sort_values('日期')
        if len(sd) >= SEQUENCE_LENGTH:
            sequences[0, i] = sd[feature_columns].values[-SEQUENCE_LENGTH:].astype(np.float32)
            valid_mask[i] = True

    seq_t = torch.FloatTensor(sequences).to(device)

    # Ensemble: average across all experts from both models
    all_model_scores = []
    for models in [ema_models, hybrid_models]:
        mc_scores_list = []
        for _ in range(MC_PASSES):
            pass_scores = []
            for model in models:
                with torch.no_grad():
                    pred = model(seq_t)
                    if isinstance(pred, tuple): pred = pred[0]
                    pass_scores.append(pred[0].cpu().numpy())
            mc_scores_list.append(np.mean(pass_scores, axis=0))
        all_model_scores.append(np.mean(mc_scores_list, axis=0))
    mc_scores = np.mean(all_model_scores, axis=0)

    raw_scores = {}
    for i, sid in enumerate(stock_ids):
        raw_scores[sid] = float(mc_scores[i]) if valid_mask[i] else -float('inf')

    data = raw_data[raw_data['日期'] <= actual_pred_date]
    filtered_ids = volatility_filter(data, stock_ids, pd.Timestamp(actual_pred_date).strftime('%Y-%m-%d'), top_pct=0.95)
    bounce_flags = bounce_confirm(data, filtered_ids, pd.Timestamp(actual_pred_date).strftime('%Y-%m-%d'))
    quality_scores = compute_quality_score(data, filtered_ids, pd.Timestamp(actual_pred_date).strftime('%Y-%m-%d'))

    final_scores = {}
    for sid in filtered_ids:
        score = raw_scores.get(sid, -float('inf'))
        if sid not in bounce_flags:
            score *= 0.92
        quality = quality_scores.get(sid, 0.5)
        score += (quality - 0.5) * 0.05
        final_scores[sid] = score

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top5_ids = [sid for sid, _ in ranked[:5]]
    selected, weights = equal_weight_allocate(top5_ids)
    picks = list(zip(selected, weights))

    # Find T+1 and T+5 dates
    all_dates_after = [d for d in all_trade_dates if d > actual_pred_date]
    if len(all_dates_after) < 5:
        print(f'  {actual_pred_date.date()}: not enough future dates')
        continue
    t1_date = all_dates_after[0]
    t5_date = all_dates_after[min(4, len(all_dates_after) - 1)]

    week_rets = []
    for sid, weight in picks:
        t1_data = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t1_date)]
        t5_data = full_df[(full_df['股票代码'] == sid) & (full_df['日期'] == t5_date)]
        if len(t1_data) == 0 or len(t5_data) == 0:
            r = 0.0
        else:
            r = (float(t5_data.iloc[0]['开盘']) - float(t1_data.iloc[0]['开盘'])) / float(t1_data.iloc[0]['开盘'])
        week_rets.append(r * weight)
    wr = sum(week_rets)
    sids = [p[0] for p in picks]
    results.append({'week': pred_date_str, 'stocks': sids, 'return': wr})
    print(f'  {pred_date_str} (predict={actual_pred_date.date()}): {sids} | {wr*100:+.2f}%')

# Summary
print(f'\n{"="*55}')
print(f'{"Week":<15} {"Return":>8}')
print('-' * 30)
total = 0
for r in results:
    print(f'{r["week"]:<15} {r["return"]*100:>+7.2f}%')
    total += r["return"]
if results:
    print(f'{"Total":<15} {total*100:>+7.2f}%')
    print(f'{"Avg Weekly":<15} {(total/len(results))*100:>+7.2f}%')
print('Done!')
