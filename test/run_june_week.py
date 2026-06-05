"""
6月第一周 (6/1-6/5): 下载最新数据 → V1/V3预测 → 计算收益
预测截止日: 2026-05-29 (6月前最后交易日)
评估区间: 2026-06-01 ~ 2026-06-05
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
import numpy as np, pandas as pd, torch, joblib
from tqdm import tqdm
from collections import Counter

from config_v3 import *
from ensemble_models import StockTransformerExpert, ConvStockExpert, MetaAggregator, MonthSeasonalExpert
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from risk_filter import compute_risk_scores, apply_risk_filter

BASE = os.path.join(os.path.dirname(__file__), '..')
TRAIN_PATH = os.path.join(BASE, 'data', 'train.csv')
TEST_PATH = os.path.join(BASE, 'data', 'test.csv')
HS300_PATH = os.path.join(BASE, 'data', 'hs300_stock_list.csv')
V1_DIR = os.path.join(BASE, 'model', 'v1_ensemble')
V3_DIR = os.path.join(BASE, 'model', 'v2_ensemble')
JUNE_NEW = os.path.join(BASE, 'data', 'june_new.csv')

CUTOFF_DATE = '2026-05-29'
EVAL_START = '2026-06-01'
EVAL_END = '2026-06-05'


def _engineer_158plus39(df):
    return engineer_features_158plus39(df, add_market=False)
feature_engineer_func_map['158+39'] = _engineer_158plus39


def download_missing_data():
    """下载 2026-06-04 ~ 06-05 的数据 (test.csv 已有 05-28~06-03)"""
    try:
        import baostock as bs
    except ImportError:
        print("baostock 未安装，跳过下载。尝试用 pip install baostock")
        return None

    # 检查是否已有新数据
    if os.path.exists(JUNE_NEW):
        existing = pd.read_csv(JUNE_NEW, dtype={'股票代码': str})
        if len(existing) > 0:
            print(f"已有缓存数据 {JUNE_NEW}: {len(existing)} 行")
            return existing

    hs300 = pd.read_csv(HS300_PATH)
    codes = sorted(hs300['code'].unique())
    print(f"下载 {len(codes)} 只股票 2026-06-04 ~ 06-05 数据...")

    bs.login()
    all_rows = []
    for idx, bs_code in enumerate(tqdm(codes)):
        real = bs_code.replace('sh.', '').replace('sz.', '').zfill(6)
        try:
            rs = bs.query_history_k_data_plus(bs_code,
                'date,open,high,low,close,volume,amount,amplitude,pctChg,turn',
                start_date='2026-06-04', end_date='2026-06-05',
                frequency='d', adjustflag='1')
            while (rs.error_code == '0') & rs.next():
                d = rs.get_row_data()
                if d[0]:
                    all_rows.append({
                        '股票代码': real,
                        '日期': d[0],
                        '开盘': float(d[1]) if d[1] else None,
                        '最高': float(d[2]) if d[2] else None,
                        '最低': float(d[3]) if d[3] else None,
                        '收盘': float(d[4]) if d[4] else None,
                        '成交量': float(d[5]) if d[5] else 0,
                        '成交额': float(d[6]) if d[6] else 0,
                        '振幅': float(d[7]) if d[7] else 0,
                        '涨跌幅': float(d[8]) if d[8] else 0,
                        '换手率': float(d[9]) if d[9] else 0,
                    })
        except:
            pass
    bs.logout()

    if not all_rows:
        print("未下载到任何数据 (可能 baostock 尚未更新今日数据)")
        return None

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=['开盘', '收盘'])  # 去掉停牌
    df['涨跌额'] = 0
    df['前收盘'] = df['收盘'] - df['涨跌额']
    # 补全 test.csv 有的列
    for col in ['涨跌额', '前收盘']:
        if col not in df.columns:
            df[col] = 0
    df.to_csv(JUNE_NEW, index=False)
    print(f"下载完成: {len(df)} 行, {df['股票代码'].nunique()} 只股票, "
          f"日期: {sorted(df['日期'].unique())}")
    return df


def build_full_dataset():
    """合并 train + test + 新数据"""
    train = pd.read_csv(TRAIN_PATH, dtype={'股票代码': str})
    test = pd.read_csv(TEST_PATH, dtype={'股票代码': str})
    train['股票代码'] = train['股票代码'].astype(str).str.zfill(6)
    test['股票代码'] = test['股票代码'].astype(str).str.zfill(6)

    # 确保列一致
    cols = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅']
    train = train[cols].copy()

    new = download_missing_data()
    if new is not None:
        new = new[cols].copy()
        full = pd.concat([train, test[cols], new], ignore_index=True)
    else:
        full = pd.concat([train, test[cols]], ignore_index=True)

    full['日期'] = pd.to_datetime(full['日期'])
    full = full.drop_duplicates(subset=['股票代码', '日期']).sort_values(['股票代码', '日期']).reset_index(drop=True)
    print(f"全量数据: {len(full)} 行, {full['日期'].min().date()} ~ {full['日期'].max().date()}")
    return full


def preprocess_for_date(df, stockid2idx):
    feature_engineer = feature_engineer_func_map[FEATURE_NUM]
    feature_columns = feature_cloums_map[FEATURE_NUM]
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    processed = pd.concat([feature_engineer(g) for g in tqdm(groups, desc='  特征工程', leave=False)]).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])
    return processed, feature_columns


def build_sequences(data, features, stock_ids, target_date):
    sequences, seq_stock_ids = [], []
    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= target_date)]
        hist = hist.sort_values('日期').tail(SEQUENCE_LENGTH)
        if len(hist) == SEQUENCE_LENGTH:
            sequences.append(hist[features].values.astype(np.float32))
            seq_stock_ids.append(sid)
    return np.asarray(sequences, dtype=np.float32) if sequences else np.array([]), seq_stock_ids


def load_v1_experts(feature_dim, num_stocks, device):
    with open(os.path.join(V1_DIR, 'ensemble_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    experts = []
    for ec in cfg['expert_configs']:
        t = ec.get('type', 'transformer')
        if t == 'transformer':
            m = StockTransformerExpert(feature_dim, ec, num_stocks)
        elif t == 'conv':
            m = ConvStockExpert(feature_dim, ec, num_stocks)
        elif t == 'month_seasonal':
            m = MonthSeasonalExpert(feature_dim, ec, num_stocks)
        else:
            continue
        path = os.path.join(V1_DIR, f'expert_{ec["name"]}.pth')
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
            m.to(device)
            experts.append(m)
    meta = MetaAggregator(len(experts), num_stocks, hidden_dim=64)
    meta_path = os.path.join(V1_DIR, 'meta_aggregator.pth')
    if os.path.exists(meta_path):
        meta.load_state_dict(torch.load(meta_path, map_location=device))
    meta.to(device)
    return experts, meta


def v1_predict(experts, meta, x, device):
    all_scores = []
    for expert in experts:
        expert.train()
        mc = []
        with torch.no_grad():
            for _ in range(20):
                mc.append(expert(x).squeeze(0))
        all_scores.append(torch.stack(mc).mean(dim=0))
    stack = torch.stack(all_scores, dim=-1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        fused = meta(stack).squeeze(0).cpu().numpy()
    return fused


def v3_predict(experts, weights, x, device, seq_ids, risk_scores, stress, risk_label='默认'):
    NUM_ROUNDS = 5
    MC_SPR = 30
    chunk_size = MAX_STOCKS_PER_CHUNK if device.type == 'cuda' else 9999
    use_amp = USE_AMP and device.type == 'cuda'

    # 根据风险标签选参数
    risk_params = {
        '保守': (80, 1, 5),
        '适中': (85, 3, 5),
        '宽松': (90, 3, 5),
        '无过滤': (100, 5, 5),
    }
    max_risk, min_pos, max_pos = risk_params.get(risk_label, (90, 3, 5))

    all_top5 = []
    for r in range(NUM_ROUNDS):
        torch.manual_seed(42 + r * 100)
        np.random.seed(42 + r * 100)
        rnd_scores = []
        for expert in experts:
            expert.train()
            mc = []
            with torch.no_grad():
                for _ in range(MC_SPR):
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
        sel, _ = apply_risk_filter(fused, seq_ids, risk_scores, stress,
                                   max_risk_score=max_risk, min_positions=min_pos, max_positions=max_pos)
        all_top5.extend(sel)

    vc = Counter(all_top5)
    consensus = [s for s, c in vc.most_common() if c >= 3]
    if len(consensus) < 1:
        consensus = [s for s, _ in vc.most_common(3)]
    return consensus


def calc_weighted_return(stock_ids, weights, eval_data):
    """计算评估期内的加权收益，用首日和末日开盘价"""
    total = 0.0
    for sid, w in zip(stock_ids, weights):
        stock = eval_data[eval_data['股票代码'] == sid].sort_values('日期')
        if len(stock) >= 2:
            start_open = stock.iloc[0]['开盘']
            end_open = stock.iloc[-1]['开盘']
            ret = (end_open - start_open) / start_open
            total += w * ret
    return total


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"设备: {device}")

    # 1. 准备全量数据
    print("\n" + "=" * 50)
    print("Step 1: 准备数据")
    full_df = build_full_dataset()
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(stock_ids)}
    num_stocks = len(stock_ids)

    # 评估期数据
    eval_data = full_df[(full_df['日期'] >= EVAL_START) & (full_df['日期'] <= EVAL_END)].copy()
    eval_dates = sorted(eval_data['日期'].unique())
    print(f"评估期: {eval_dates[0]} ~ {eval_dates[-1]}, {len(eval_dates)} 天 "
          f"({eval_data['股票代码'].nunique()} 只股票)")

    # 2. 预处理 (截止 cutoff_date)
    print("\nStep 2: 特征工程 (截止 {})".format(CUTOFF_DATE))
    train_df = full_df[full_df['日期'] <= CUTOFF_DATE].copy()
    processed, features = preprocess_for_date(train_df, stockid2idx)
    feature_dim = len(features)

    # 风险评分
    risk_raw, stress = compute_risk_scores(
        processed, features, stock_ids, stock_ids, train_df['日期'].max())
    print(f"市场压力: {stress:.1%}")

    # 标准化
    scaler = joblib.load(os.path.join(V3_DIR, 'scaler.pkl'))
    common = [c for c in scaler.feature_names_in_ if c in processed.columns]
    processed[common] = scaler.transform(processed[common])

    # 构建序列
    pred_date = pd.to_datetime(CUTOFF_DATE)
    seq_np, seq_ids = build_sequences(processed, common, stock_ids, pred_date)
    print(f"可用股票: {len(seq_ids)}")

    risk_scores = {sid: risk_raw.get(sid, 50) for sid in seq_ids}
    x = torch.from_numpy(seq_np).unsqueeze(0).to(device)

    # 3. 加载模型
    print("\nStep 3: 加载模型")
    print("V1...")
    v1_experts, v1_meta = load_v1_experts(feature_dim, num_stocks, device)
    print(f"  {len(v1_experts)} experts + MetaAggregator")

    print("V3...")
    v3_models = []
    v3_names = ['balanced_v2', 'deep_v2', 'conv_multiscale', 'conv_deep']
    v3_weights_raw = [0.1855, 0.1215, 0.1113, 0.0804]
    v3_w = [w / sum(v3_weights_raw) for w in v3_weights_raw]
    for name in v3_names:
        path = os.path.join(V3_DIR, f'expert_{name}.pth')
        if name.startswith('conv'):
            cfg = {'name': name, 'type': 'conv',
                   'hidden_channels': 256 if 'multi' in name else 384,
                   'nhead': 4, 'dropout': 0.12 if 'multi' in name else 0.15,
                   'mc_dropout_rate': 0.1 if 'multi' in name else 0.12,
                   'sd_prob': 0.9 if 'multi' in name else 0.85}
            m = ConvStockExpert(feature_dim, cfg, num_stocks)
        else:
            cfg = {'name': name, 'type': 'transformer',
                   'd_model': 256 if name == 'balanced_v2' else 192, 'nhead': 4,
                   'num_layers': 6 if name == 'balanced_v2' else 8,
                   'dim_feedforward': 512 if name == 'balanced_v2' else 384,
                   'dropout': 0.1,
                   'mc_dropout_rate': 0.1 if name == 'balanced_v2' else 0.12,
                   'sd_prob': 0.9 if name == 'balanced_v2' else 0.85}
            m = StockTransformerExpert(feature_dim, cfg, num_stocks)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        v3_models.append(m)
    print(f"  {len(v3_models)} experts + 投票共识")

    # 4. 预测
    print("\n" + "=" * 50)
    print("Step 4: 预测")

    # V1
    print("\n[V1]")
    v1_fused = v1_predict(v1_experts, v1_meta, x, device)
    v1_order = np.argsort(v1_fused)[::-1]
    v1_top5 = [seq_ids[i] for i in v1_order[:5]]
    v1_ret = calc_weighted_return(v1_top5, [0.2]*5, eval_data)

    hs300 = pd.read_csv(HS300_PATH)
    hs300['code'] = hs300['code'].str.replace('sh.','').str.replace('sz.','').str.zfill(6)
    code_name = dict(zip(hs300['code'], hs300['code_name']))

    print(f"  Top5: {v1_top5}")
    for sid in v1_top5:
        s = eval_data[eval_data['股票代码'] == sid].sort_values('日期')
        if len(s) >= 2:
            r = (s.iloc[-1]['开盘'] - s.iloc[0]['开盘']) / s.iloc[0]['开盘']
            print(f"    {sid} {code_name.get(sid,'?')}: {r:+.4%}")
    print(f"  V1 加权收益: {v1_ret:+.4%}")

    # V3 (多风险参数)
    print("\n[V3]")
    v3_all = {}
    for label in ['宽松', '无过滤']:
        v3_top = v3_predict(v3_models, v3_w, x, device, seq_ids, risk_scores, stress, label)
        if len(v3_top) > 5:
            v3_top = v3_top[:5]
        w_list = [1.0/len(v3_top)] * len(v3_top) if v3_top else []
        v3_ret = calc_weighted_return(v3_top, w_list, eval_data) if v3_top else 0
        v3_all[label] = (v3_top, v3_ret)
        print(f"  [{label}] Top{len(v3_top)}: {v3_top}")
        for sid in v3_top:
            s = eval_data[eval_data['股票代码'] == sid].sort_values('日期')
            if len(s) >= 2:
                r = (s.iloc[-1]['开盘'] - s.iloc[0]['开盘']) / s.iloc[0]['开盘']
                print(f"    {sid} {code_name.get(sid,'?')}: {r:+.4%}")
        print(f"  [{label}] V3 加权收益: {v3_ret:+.4%}")

    # 5. 汇总
    print("\n" + "=" * 60)
    print("6月第一周 (6/1-6/5) 收益对比")
    print("=" * 60)
    print(f"  预测截止: {CUTOFF_DATE}, 评估: {eval_dates[0]} ~ {eval_dates[-1]}")
    print(f"  评估天数: {len(eval_dates)}")
    print()
    print(f"  {'':<12} {'选股':<35} {'收益'}")
    print(f"  {'V1':<12} {','.join(v1_top5):<35} {v1_ret:+.4%}")
    for label, (stocks, ret) in v3_all.items():
        print(f"  {'V3-'+label:<12} {','.join(stocks):<35} {ret:+.4%}")

    best_label = max(v3_all, key=lambda k: v3_all[k][1])
    best_v3 = v3_all[best_label][1]
    print(f"\n  V1 vs V3最佳: {best_v3 - v1_ret:+.4%}")

    # 保存 result.csv
    output_dir = os.path.join(BASE, 'output')
    os.makedirs(output_dir, exist_ok=True)
    v3_best_stocks = v3_all[best_label][0]
    v3_best_w = [1.0/len(v3_best_stocks)] * len(v3_best_stocks) if v3_best_stocks else []
    result = pd.DataFrame({'stock_id': v3_best_stocks, 'weight': v3_best_w})
    result.to_csv(os.path.join(output_dir, 'result.csv'), index=False)
    print(f"\n已保存 output/result.csv (V3-{best_label})")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
