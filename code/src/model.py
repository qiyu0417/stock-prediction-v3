"""
优化版 StockTransformer 模型
- PositionalEncoding: 正弦位置编码
- FeatureAttention: 时序特征注意力聚合
- StockEmbeddedCrossAttention: 带可学习股票嵌入的交叉注意力
- StockTransformer: 排序学习主模型 + 多任务回归头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeatureAttention(nn.Module):
    """特征注意力模块：对时间维做注意力加权聚合"""
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        return self.dropout(attended)


class StockEmbeddedCrossAttention(nn.Module):
    """
    带可学习股票嵌入的交叉注意力模块。
    每只股票维护一个 d_model 维的可学习嵌入向量，
    在 Query 端注入股票身份信息，帮助模型隐式学习板块/风格分组。
    """
    def __init__(self, d_model, num_stocks, nhead=4, dropout=0.1):
        super(StockEmbeddedCrossAttention, self).__init__()
        self.stock_embed = nn.Embedding(num_stocks, d_model)
        nn.init.normal_(self.stock_embed.weight, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features, stock_indices=None):
        # stock_features: [batch, num_stocks, d_model]
        # stock_indices: [batch, num_stocks] 股票ID，用于查找嵌入
        batch_size, num_stocks, d_model = stock_features.shape

        if stock_indices is not None:
            # 获取每只股票的可学习嵌入并加到 Query 上
            embeds = self.stock_embed(stock_indices)  # [B, N, d_model]
            query = stock_features + embeds
        else:
            query = stock_features

        attended, _ = self.cross_attention(query, stock_features, stock_features)
        output = self.norm(stock_features + self.dropout(attended))
        return output


class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks):
        super(StockTransformer, self).__init__()
        self.model_type = 'StockTransformerOptimized'
        self.config = config
        self.num_stocks = num_stocks

        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(
            config['d_model'], config['dropout'], config['sequence_length']
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config['num_layers']
        )

        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])

        self.cross_stock_attention = StockEmbeddedCrossAttention(
            config['d_model'], num_stocks, config['nhead'], config['dropout']
        )

        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )

        # 排序分数头
        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )

        # 多任务回归头 —— 预测绝对收益率
        self.return_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src, stock_indices=None, return_features=False):
        # src: [batch, num_stocks, seq_len, feature_dim]
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        # 安全校验：替换 NaN/Inf
        if torch.isnan(src).any() or torch.isinf(src).any():
            src = torch.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)

        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)

        temporal_features = self.temporal_encoder(src_proj)
        aggregated_features = self.feature_attention(temporal_features)

        stock_features = aggregated_features.view(batch_size, num_stocks, -1)

        # 安全校验 stock_indices
        if stock_indices is not None:
            stock_indices = stock_indices.clamp(0, self.num_stocks - 1)

        interactive_features = self.cross_stock_attention(stock_features, stock_indices)
        interactive_features = interactive_features.view(batch_size * num_stocks, -1)

        ranking_features = self.ranking_layers(interactive_features)

        scores = self.score_head(ranking_features)
        scores = scores.view(batch_size, num_stocks)

        if return_features:
            return scores, ranking_features

        # 多任务：同时输出排序分数和收益率预测
        pred_returns = self.return_head(ranking_features)
        pred_returns = pred_returns.view(batch_size, num_stocks)

        return scores, pred_returns
