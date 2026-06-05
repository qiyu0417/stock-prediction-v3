"""
DeepSleep V1 — 沪深300排序选股模型

4专家集成 (Transformer + Conv) × 4D市场状态 × 反弹确认 × 质量加分 × 置信度分配

使用方法:
    python code/src/deepsleep_v1.py                          # 预测 test.csv 日期
    python code/src/deepsleep_v1.py --date 2026-06-06        # 预测指定日期
    python code/src/deepsleep_v1.py --train                  # 重新训练
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

from config_v5 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from market_regime import compute_market_regime
from quality_filter import (
    bounce_confirm, compute_quality_score,
    confidence_weighted_allocate, volatility_filter
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'v5_ensemble')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)


feature_engineer_func_map['158+39'] = _engineer_158plus39


def load_experts(feature_dim, num_stocks, device):
    """加载 DeepSleep V1 的 4 个专家模型"""
    with open(os.path.join(MODEL_DIR, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    models = []
    for ec in cfg['expert_configs']:
        path = os.path.join(MODEL_DIR, f'expert_{ec["name"]}.pth')
        if not os.path.exists(path):
            continue
        if ec['type'] == 'transformer':
            m = StockTransformerExpert(feature_dim, ec, num_stocks)
        elif ec['type'] == 'conv':
            m = ConvStockExpert(feature_dim, ec, num_stocks)
        else:
            continue
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        models.append(m)
    return models, [1.0 / len(models)] * len(models)


def mc_inference(experts, weights, x, device):
    """MC Dropout 推理 — 5轮 × 30次采样 → 平均评分"""
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'

    all_fused = []
    for r in range(5):
        torch.manual_seed(42 + r * 100)
        np.random.seed(42 + r * 100)
        rnd_scores = []
        for expert in experts:
            expert.train()  # 保持 dropout 开启
            mc = []
            with torch.no_grad():
                for _ in range(30):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if x.size(1) <= chunk_size:
                            s = expert(x).squeeze(0)
                        else:
                            cs = []
                            for start in range(0, x.size(1), chunk_size):
                                end = min(start + chunk_size, x.size(1))
                                cs.append(expert(x[:, start:end].contiguous()).squeeze(0))
                            s = torch.cat(cs, dim=0)
                    mc.append(s)
            rnd_scores.append(torch.stack(mc).mean(dim=0).cpu().numpy())
        fused = np.zeros(len(rnd_scores[0]))
        for w, sc in zip(weights, rnd_scores):
            fused += w * sc
        all_fused.append(fused)
    return np.mean(all_fused, axis=0)


def preprocess(df, sid2idx, winsor_bounds, scaler):
    """特征工程 + Winsorization + 标准化"""
    fe = _engineer_158plus39
    fc = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    p = pd.concat([fe(g) for g in tqdm(groups, desc='特征工程', leave=False)]).reset_index(drop=True)
    p['instrument'] = p['股票代码'].map(sid2idx)
    p = p.dropna(subset=['instrument']).copy()
    p['instrument'] = p['instrument'].astype(np.int64)
    p['日期'] = pd.to_datetime(p['日期'])
    for col, (lo, hi) in winsor_bounds.items():
        if col in p.columns:
            p[col] = p[col].clip(lo, hi)
    common = [c for c in scaler.feature_names_in_ if c in p.columns]
    p[common] = scaler.transform(p[common])
    return p, common


def build_sequences(data, features, stock_ids, target_date):
    """为每只股票构建最近 SEQUENCE_LENGTH 天的序列"""
    seqs, sids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= target_date)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            seqs.append(hist[features].values.astype(np.float32))
            sids.append(sid)
    return np.asarray(seqs, dtype=np.float32) if seqs else np.array([]), sids


def predict(experts, weights, x, device, seq_ids, proc_data, all_stocks, ref_date):
    """
    DeepSleep V1 完整预测管线:

    Step 1: MC Dropout → 原始评分
    Step 2: 4D 市场状态检测 → 极端风险日空仓
    Step 3: 波动率过滤 → 移除 top 5% 高波动股
    Step 4: 反弹确认 → 近2日 < 0.8% 的股票降低评分
    Step 5: 质量加分 → 稳健趋势股获得加成
    Step 6: 置信度不等权分配 → softmax(t=0.3) + 30% 单票上限
    """
    # Step 1: 原始评分
    raw = mc_inference(experts, weights, x, device)
    score_map = {sid: float(raw[i]) for i, sid in enumerate(seq_ids) if i < len(raw)}

    # Step 2: 市场状态检测
    regime = compute_market_regime(proc_data, [], all_stocks, pd.to_datetime(ref_date))
    if regime.get('skip_trading'):
        return [], [], regime

    # Step 3: 波动率过滤
    fids = volatility_filter(proc_data, seq_ids, ref_date, top_pct=0.95)
    if len(fids) < 3:
        fids = seq_ids[:10]

    # Step 4: 反弹确认
    confirmed = bounce_confirm(proc_data, fids, ref_date)
    for sid in fids:
        if sid not in confirmed:
            score_map[sid] = score_map.get(sid, 0) * 0.92

    # Step 5: 质量加分
    quality = compute_quality_score(proc_data, fids, ref_date)
    for sid in fids:
        if sid in score_map and sid in quality:
            score_map[sid] += (quality[sid] - 0.5) * 0.05

    # Step 6: 不等权分配
    sorted_s = sorted(fids, key=lambda s: score_map.get(s, -999), reverse=True)
    sel, w = confidence_weighted_allocate(
        score_map, sorted_s, {},
        max_positions=5, temperature=0.3,
        max_single=0.30, use_sigma=False
    )
    return sel, w, regime


def main():
    parser = argparse.ArgumentParser(description='DeepSleep V1 预测')
    parser.add_argument('--date', type=str, default=None,
                        help='预测日期 (YYYY-MM-DD), 默认使用 test.csv 的日期')
    parser.add_argument('--data', type=str, default=os.path.join(DATA_DIR, 'train.csv'),
                        help='训练/历史数据路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径, 默认 output/result.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DeepSleep V1 | Device: {device}")

    # 加载数据
    full = pd.read_csv(args.data, dtype={'股票代码': str})
    full['股票代码'] = full['股票代码'].astype(str).str.zfill(6)
    full['日期'] = pd.to_datetime(full['日期'])
    all_stocks = sorted(full['股票代码'].unique())
    fdim = len(feature_cloums_map[FEATURE_NUM])
    nstocks = len(all_stocks)

    # 加载模型 + 预处理器
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    with open(os.path.join(MODEL_DIR, 'winsor_bounds.json')) as f:
        winsor = json.load(f)
    experts, ew = load_experts(fdim, nstocks, device)
    print(f"加载 {len(experts)} 个专家: {[e.__class__.__name__ for e in experts]}")

    # 确定预测日期
    if args.date:
        target_date = args.date
        ref_date = full[full['日期'] < target_date]['日期'].max()
    else:
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), dtype={'股票代码': str})
        test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
        target_date = str(test_df['日期'].iloc[0])
        ref_date = str(full['日期'].max())

    print(f"预测日期: {target_date}, 参考截止: {ref_date}")

    # 预处理
    train_df = full[full['日期'] <= ref_date].copy()
    sid2idx = {s: i for i, s in enumerate(all_stocks)}
    proc, cols = preprocess(train_df, sid2idx, winsor, scaler)
    seqs, ids = build_sequences(proc, cols, all_stocks, pd.to_datetime(ref_date))
    x = torch.from_numpy(seqs).unsqueeze(0).to(device)

    # 预测
    stocks, wts, regime = predict(experts, ew, x, device, ids, proc, all_stocks, ref_date)

    market_status = regime.get('regime', 'unknown')
    composite = regime.get('composite', 0)
    print(f"市场状态: {market_status} (压力={composite:.2f})")

    if regime.get('skip_trading'):
        print("触发空仓信号 — 不推荐建仓")
    else:
        print(f"DeepSleep V1 Top{len(stocks)}:")
        for i, (s, w) in enumerate(zip(stocks, wts)):
            print(f"  {i + 1}. {s} — {w:.1%}")

    # 输出
    out_path = args.output or os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'result.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = pd.DataFrame({'股票代码': stocks, '权重': wts})
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至 {out_path}")


if __name__ == '__main__':
    main()
