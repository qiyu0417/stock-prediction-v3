"""
优化版训练脚本
- 标签修复: (close_t5 - open_t1) / open_t1  消除前瞻偏差
- 市场特征: 201维 (158+39+4市场)
- 多任务学习: 排序损失 + 回归损失
- 训练策略: OneCycleLR + 早停 + batch_size=8
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized, CROSS_SECTIONAL_FEATURES, INDUSTRY_COLS
import joblib
import os
import json
import multiprocessing as mp
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------- 特征列定义 ----------

_MARKET_COLS = ['market_return', 'market_up_ratio', 'market_volume_sum', 'market_volatility']

_BASE_COLS_39 = [
    'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
]

_ALPHA_158_COLS = [
    'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
    'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
    'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
    'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
    'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
    'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
    'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60',
    'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60',
    'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
    'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60',
    'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60',
    'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60',
    'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60',
    'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60',
    'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
    'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60',
    'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
    'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60',
    'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
    'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60',
    'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
    'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60',
    'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
    'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60',
    'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
    'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60',
    'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
    'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60',
    'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60',
    'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60',
    'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
]

_TECH_39_ONLY = [
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
    'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
    'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
    'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
]

_CS_COLS = [f'CS_{f}' for f in CROSS_SECTIONAL_FEATURES]
_INDUSTRY_COLS = INDUSTRY_COLS

feature_cloums_map = {
    '39': _BASE_COLS_39,
    '158+39': _ALPHA_158_COLS + _TECH_39_ONLY,
    '158+39+market': _ALPHA_158_COLS + _TECH_39_ONLY + _MARKET_COLS,
    '158+39+CS': _ALPHA_158_COLS + _TECH_39_ONLY + _CS_COLS,
    '158+39+industry': _ALPHA_158_COLS + _TECH_39_ONLY + _INDUSTRY_COLS,
    '158+39+market+industry': _ALPHA_158_COLS + _TECH_39_ONLY + _MARKET_COLS + _INDUSTRY_COLS,
}

def _engineer_158plus39_market(df):
    return engineer_features_158plus39(df, add_market=True)


def _engineer_158plus39_industry(df):
    return engineer_features_158plus39(df, add_market=False, add_industry=True)


def _engineer_158plus39_market_industry(df):
    return engineer_features_158plus39(df, add_market=True, add_industry=True)


feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
    '158+39+market': _engineer_158plus39_market,
    '158+39+industry': _engineer_158plus39_industry,
    '158+39+market+industry': _engineer_158plus39_market_industry,
}


# ---------- 标签构建 ----------

def _build_label_and_clean(processed, drop_small_open=True):
    """构建标签: (open_T+5 - open_T+1) / open_T+1 —— 比赛公式 (买入T+1开盘, 卖出T+5开盘)
    ⛔ 必须用开盘价！2026-06-19 修复：close_t5 → open_t5
    """
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t5', 'open_t1'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, \
        f"Unsupported feature_num: {config['feature_num']}"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups),
                                   total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# ---------- 损失函数 ----------

class WeightedRankingLoss(nn.Module):
    """组合的加权排序损失函数 + 多任务回归损失"""
    def __init__(self, temperature=1.0, k=5, weight_factor=2.0,
                 pairwise_weight=1.0, base_weight=1.0, multitask_weight=0.3):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight
        self.multitask_weight = multitask_weight
        self.mse = nn.MSELoss()

    def listwise_loss(self, y_pred, y_true, weights):
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        return (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()

    def pairwise_loss(self, y_pred, y_true, weights):
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        pairwise = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        weighted_loss = pairwise * mask * weight_matrix
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        return (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()

    def forward(self, y_pred, y_true, pred_returns=None, true_returns=None):
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        _, top_indices = torch.topk(y_true, k, dim=1)
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor

        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        ranking_loss = listwise + self.pairwise_weight * pairwise

        if pred_returns is not None and true_returns is not None:
            reg_loss = self.mse(pred_returns, true_returns)
            return ranking_loss + self.multitask_weight * reg_loss

        return ranking_loss


class LambdaRankLoss(nn.Module):
    """LambdaRank (optimized): only Top-K pairs, O(K×n) instead of O(n²)"""
    def __init__(self, k=5, sigma=1.0):
        super().__init__()
        self.k = k
        self.sigma = sigma

    def forward(self, y_pred, y_true):
        batch, n = y_pred.shape
        k = min(self.k, n)
        device = y_pred.device

        positions = torch.arange(1, n + 1, device=device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1)  # [n]

        total_loss = torch.tensor(0.0, device=device)
        for b in range(batch):
            # Top-K by true return
            _, top_k_idx = torch.topk(y_true[b], k)

            # Score diff: top-K vs ALL items  [k, n]
            s_diff = y_pred[b, top_k_idx].unsqueeze(1) - y_pred[b].unsqueeze(0)

            # True sign: should top-K score higher/lower than others
            y_sign = torch.sign(y_true[b, top_k_idx].unsqueeze(1) - y_true[b].unsqueeze(0))

            # NDCG position discount weight  [k, n]
            d_top = discounts[top_k_idx].unsqueeze(1)  # [k, 1]
            d_all = discounts.unsqueeze(0)  # [1, n]
            dcg_w = torch.abs(d_top - d_all)  # [k, n]
            dcg_w = dcg_w / (dcg_w.sum() + 1e-8)

            pair_loss = torch.log(1 + torch.exp(-self.sigma * s_diff * y_sign))
            total_loss += (pair_loss * dcg_w).sum()

        return total_loss / batch


class ListMLELoss(nn.Module):
    """ListMLE: Plackett-Luce 排列概率, 聚焦 Top-K 排序质量"""
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        batch, n = y_pred.shape
        k = min(self.k, n)
        _, idx = torch.sort(y_true, dim=1, descending=True)
        losses = []
        for b in range(batch):
            for pos in range(k):
                remaining = idx[b, pos:]
                log_denom = torch.logsumexp(y_pred[b, remaining], dim=0)
                losses.append(log_denom - y_pred[b, idx[b, pos]])
        return torch.stack(losses).mean()


class ApproxNDCGLoss(nn.Module):
    """可微分 NDCG: sigmoid 近似排序, 期望 rank 折现"""
    def __init__(self, k=5, temperature=0.5):
        super().__init__()
        self.k = k
        self.temperature = temperature

    def _approx_ranks(self, scores):
        s_i = scores.unsqueeze(-1)
        s_j = scores.unsqueeze(-2)
        return 1 + torch.sigmoid((s_j - s_i) / self.temperature).sum(dim=-1)

    def forward(self, y_pred, y_true):
        batch, n = y_pred.shape
        device = y_pred.device
        k = min(self.k, n)

        expected_ranks = self._approx_ranks(y_pred)
        discounts = 1.0 / torch.log2(expected_ranks + 1)
        approx_dcg = (y_true * discounts).sum(dim=-1)

        positions = torch.arange(1, k + 1, device=device, dtype=torch.float32)
        ideal_discounts = 1.0 / torch.log2(positions + 1)
        _, ideal_idx = torch.topk(y_true, k, dim=-1)
        ideal_dcg = (y_true.gather(-1, ideal_idx) * ideal_discounts.unsqueeze(0)).sum(dim=-1)

        ndcg = approx_dcg / (ideal_dcg + 1e-8)
        return -ndcg.mean()


def _approx_ndcg_loss(y_pred, y_true, k=5, temperature=0.5):
    batch, n = y_pred.shape
    device = y_pred.device
    k = min(k, n)

    s_i = y_pred.unsqueeze(-1)
    s_j = y_pred.unsqueeze(-2)
    expected_ranks = 1 + torch.sigmoid((s_j - s_i) / temperature).sum(dim=-1)

    discounts = 1.0 / torch.log2(expected_ranks + 1)
    approx_dcg = (y_true * discounts).sum(dim=-1)

    positions = torch.arange(1, k + 1, device=device, dtype=torch.float32)
    ideal_discounts = 1.0 / torch.log2(positions + 1)
    _, ideal_idx = torch.topk(y_true, k, dim=-1)
    ideal_dcg = (y_true.gather(-1, ideal_idx) * ideal_discounts.unsqueeze(0)).sum(dim=-1)

    ndcg = approx_dcg / (ideal_dcg + 1e-8)
    return -ndcg.mean()


class HybridRankingLoss(nn.Module):
    """ListMLE + ApproxNDCG + LambdaRank 混合, 聚焦 Top-5"""
    def __init__(self, k=5, listmle_weight=1.0, ndcg_weight=0.5,
                 lambda_weight=0.3, sigma=1.0):
        super().__init__()
        self.k = k
        self.listmle_weight = listmle_weight
        self.ndcg_weight = ndcg_weight
        self.lambda_weight = lambda_weight
        self.listmle = ListMLELoss(k=k)
        self.lambdarank = LambdaRankLoss(k=k, sigma=sigma)

    def forward(self, y_pred, y_true):
        loss = self.listmle_weight * self.listmle(y_pred, y_true)
        loss += self.ndcg_weight * _approx_ndcg_loss(y_pred, y_true, self.k)
        loss += self.lambda_weight * self.lambdarank(y_pred, y_true)
        return loss


# ---------- 评估指标 ----------

def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    batch_size = y_pred.size(0)
    pred_return_sum_list, max_return_sum_list, random_return_sum_list = [], [], []
    ratio_pred_list, ratio_random_list, final_score_list = [], [], []

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()

        random_return_sum = k * valid_true.mean().item()

        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
        'ratio_pred': np.mean(ratio_pred_list) if ratio_pred_list else 0.0,
        'ratio_random': np.mean(ratio_random_list) if ratio_random_list else 0.0,
        'final_score': np.mean(final_score_list) if final_score_list else 0.0,
    }
    return metrics


# ---------- 数据集 ----------

class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'relevance': torch.LongTensor(self.relevance_scores[idx]),
            'stock_indices': torch.LongTensor(self.stock_indices[idx])
        }


def collate_fn(batch):
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)

    padded_sequences, padded_targets, padded_relevance = [], [], []
    padded_stock_indices, masks = [], []

    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)

        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)

            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)

        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    return {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices),
        'masks': torch.stack(masks)
    }


# ---------- 训练 / 验证 ----------

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, scheduler=None):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        relevance = batch['relevance'].to(device)
        stock_indices = batch['stock_indices'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()

        scores, pred_returns = model(sequences, stock_indices)
        masked_scores = scores * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks
        masked_pred_returns = pred_returns * masks

        batch_loss = None
        batch_size = sequences.size(0)

        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            if valid_indices.numel() == 0:
                continue
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)

            valid_pred = masked_scores[i][valid_indices]
            valid_rel = masked_relevance[i][valid_indices]
            valid_ret = masked_pred_returns[i][valid_indices]
            valid_tgt = masked_targets[i][valid_indices]

            if len(valid_pred) > 1:
                loss = criterion(valid_pred.unsqueeze(0), valid_rel.unsqueeze(0),
                                 valid_ret.unsqueeze(0), valid_tgt.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += batch_loss.item()

            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_scores, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v

            local_step += 1
            if writer and local_step % 10 == 0:
                writer.add_scalar('train/loss', batch_loss.item(),
                                  global_step=epoch * len(dataloader) + local_step)
                writer.add_scalar('train/grad_norm', grad_norm,
                                  global_step=epoch * len(dataloader) + local_step)

    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step

    return total_loss / max(len(dataloader), 1), total_metrics


def evaluate_epoch(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            stock_indices = batch['stock_indices'].to(device)
            masks = batch['masks'].to(device)

            scores, pred_returns = model(sequences, stock_indices)
            masked_scores = scores * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            masked_pred_returns = pred_returns * masks

            batch_loss = None
            batch_size = sequences.size(0)

            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                if valid_indices.numel() == 0:
                    continue
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)

                valid_pred = masked_scores[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                valid_ret = masked_pred_returns[i][valid_indices]

                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true)
                    relevance_scores[sorted_indices] = torch.arange(
                        len(valid_true), 0, -1, device=device, dtype=torch.float32
                    )
                    loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0),
                                     valid_ret.unsqueeze(0), valid_true.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()

            metrics = calculate_ranking_metrics(masked_scores, masked_targets, masks, k=5)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    for k in total_metrics:
        total_metrics[k] /= max(num_batches, 1)

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)

    return avg_loss, total_metrics


# ---------- 数据划分 ----------

def split_train_val_by_last_month(df, sequence_length):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围: {val_start.date()} 到 {last_date.date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


# ---------- 主程序 ----------

def main():
    from config import config
    from model import StockTransformer
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"使用设备: {device}")

    # 1. 数据加载
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, config['sequence_length']
    )

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    # 2. 特征工程
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    # 3. 标准化
    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # 4. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = \
        create_ranking_dataset_vectorized(train_data, features, config['sequence_length'])
    val_sequences, val_targets, val_relevance, val_stock_indices = \
        create_ranking_dataset_vectorized(val_data, features, config['sequence_length'],
                                          min_window_end_date=val_start.strftime('%Y-%m-%d'))

    print(f"训练集样本数: {len(train_sequences)}, 验证集样本数: {len(val_sequences)}")
    print(f"特征维度: {len(features)}")

    # 5. DataLoader
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 6. 模型
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 7. 损失函数 + 优化器 + OneCycleLR
    criterion = WeightedRankingLoss(
        k=5, temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0),
        multitask_weight=config.get('multitask_weight', 0.3),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

    steps_per_epoch = len(train_loader)
    total_steps = config['num_epochs'] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['learning_rate'],
        total_steps=total_steps, pct_start=0.1,
        anneal_strategy='cos'
    )

    # 8. 训练循环 + 早停
    best_score = -float('inf')
    best_epoch = -1
    patience_counter = 0
    patience = config.get('early_stopping_patience', 10)

    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")

        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, scheduler
        )
        print(f"Train Loss: {train_loss:.4f}, final_score: {train_metrics.get('final_score', 0):.4f}")

        eval_loss, eval_metrics = evaluate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        print(f"Eval Loss: {eval_loss:.4f}, final_score: {eval_metrics.get('final_score', 0):.4f}")

        if writer:
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step=epoch)

        current_final_score = eval_metrics.get('final_score', 0.0)
        if current_final_score > best_score + 1e-6:
            best_score = current_final_score
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"保存最佳模型 - final_score: {best_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发！{patience} 个 epoch 未提升")
                break

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final_score: {best_score:.4f}")

    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")

    if writer:
        writer.close()

    return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 final_score: {best_score:.4f} ##########")
