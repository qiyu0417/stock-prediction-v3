"""
专家模型集合 - 支持 MC Dropout（随机灭灯 + 亮度平均）
- Transformer 专家（深度/宽度/平衡/注意力/轻量）
- 卷积专家 (TCN风格时序卷积)
- 对抗学习组件 (Gradient Reversal + 时间域分类器)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# 基础组件
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeatureAttention(nn.Module):
    """特征维度注意力聚合"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attn_weights = self.attention(x)  # [N, seq_len, 1]
        attended = torch.sum(x * attn_weights, dim=1)  # [N, d_model]
        return self.dropout(attended)


class CrossStockAttention(nn.Module):
    """股票间交互注意力"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features):
        # stock_features: [batch, num_stocks, d_model]
        attended, _ = self.cross_attn(stock_features, stock_features, stock_features)
        return self.norm(stock_features + self.dropout(attended))


class StochasticDepth(nn.Module):
    """
    随机深度（随机灭灯）: 训练时以概率 p 跳过整个子层
    推理时按 survival_prob 缩放（亮度平均）
    """
    def __init__(self, survival_prob=0.9):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x, sublayer_fn):
        if not self.training:
            # 推理模式: 始终通过，按 survival_prob 缩放
            return self.survival_prob * sublayer_fn(x)

        # 训练模式: 随机灭灯
        if torch.rand(1, device=x.device).item() > self.survival_prob:
            return x  # 跳过子层
        return sublayer_fn(x) / self.survival_prob  # 缩放补偿


class MCDropoutTransformerEncoderLayer(nn.Module):
    """
    带 MC Dropout 的 Transformer 编码层
    训练时 dropout 正常工作，推理时可强制开启用于 MC 采样
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, sd_prob=0.9):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.stochastic_depth = StochasticDepth(sd_prob)

    def forward(self, src, force_dropout=False):
        # Self-attention with stochastic depth
        def attn_fn(x):
            x2, _ = self.self_attn(x, x, x)
            return self.dropout1(x2)

        src2 = self.stochastic_depth(src, attn_fn)
        src = self.norm1(src + src2)

        # Feed-forward with stochastic depth
        def ff_fn(x):
            return self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x)))))

        src2 = self.stochastic_depth(src, ff_fn)
        src = self.norm2(src + src2)
        return src


# ============================================================
# 专家模型
# ============================================================
class StockTransformerExpert(nn.Module):
    """
    单专家模型 - 支持 MC Dropout 推理
    训练时: model.train() → dropout + stochastic depth 正常工作
    推理时: model.train() + force_mc_dropout → 多次前向平均（亮度平均）
    """
    def __init__(self, input_dim, expert_config, num_stocks):
        super().__init__()
        cfg = expert_config
        self.d_model = cfg['d_model']
        self.num_stocks = num_stocks
        self.cfg_name = cfg.get('name', 'expert')
        self.mc_dropout_rate = cfg.get('mc_dropout_rate', 0.1)
        self.sd_prob = cfg.get('sd_prob', 0.9)
        self.industry_embed_dim = cfg.get('industry_embed_dim', 0)
        self.stock_embed_dim = cfg.get('stock_embed_dim', 0)

        # 输入投影
        if self.industry_embed_dim > 0:
            tech_dim = input_dim - 14
            self.input_proj = nn.Linear(tech_dim, self.d_model)
            self.industry_compressor = nn.Linear(14, self.industry_embed_dim)
            self.industry_fusion = nn.Linear(self.d_model + self.industry_embed_dim, self.d_model)
        else:
            self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, cfg['dropout'])

        # 个股Embedding
        if self.stock_embed_dim > 0:
            self.stock_embedding = nn.Embedding(num_stocks, self.stock_embed_dim)
            self.stock_emb_proj = nn.Linear(self.stock_embed_dim, self.d_model)

        # 时序编码器（多层 + 随机深度 + MC Dropout）
        self.temporal_layers = nn.ModuleList([
            MCDropoutTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=cfg['nhead'],
                dim_feedforward=cfg['dim_feedforward'],
                dropout=cfg['dropout'],
                sd_prob=self.sd_prob,
            )
            for _ in range(cfg['num_layers'])
        ])

        # 特征注意力
        self.feature_attention = FeatureAttention(self.d_model, cfg['dropout'])

        # 股票间交互
        self.cross_stock_attention = CrossStockAttention(
            self.d_model, cfg['nhead'], cfg['dropout']
        )

        # 排序层
        self.ranking_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
        )

        # 分数头
        self.score_head = nn.Sequential(
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Dropout(cfg['dropout'] * 0.5),
            nn.Linear(self.d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src):
        # src: [batch, num_stocks, seq_len, feature_dim]
        batch_size, num_stocks, seq_len, feature_dim = src.shape

        # 展平股票维度
        src_flat = src.view(batch_size * num_stocks, seq_len, feature_dim)

        # 输入投影 + 位置编码
        if self.industry_embed_dim > 0:
            tech = src_flat[..., :-14]
            ind = src_flat[..., -14:]
            x = self.input_proj(tech)
            i = self.industry_compressor(ind)
            x = torch.cat([x, i], dim=-1)
            x = self.industry_fusion(x)
        else:
            x = self.input_proj(src_flat)
        x = self.pos_encoder(x)

        # 时序编码（带随机深度）
        for layer in self.temporal_layers:
            x = layer(x)

        # 特征注意力聚合
        x = self.feature_attention(x)  # [B*N, d_model]

        # 股票间交互
        stock_features = x.view(batch_size, num_stocks, -1)
        if self.stock_embed_dim > 0:
            stock_ids = torch.arange(num_stocks, device=src.device)
            stock_emb = self.stock_embedding(stock_ids)
            stock_emb = stock_emb.unsqueeze(0).expand(batch_size, -1, -1)
            stock_features = stock_features + self.stock_emb_proj(stock_emb)
        stock_features = self.cross_stock_attention(stock_features)
        stock_features = stock_features.view(batch_size * num_stocks, -1)

        # 排序层
        ranking_features = self.ranking_layers(stock_features)

        # 分数头
        scores = self.score_head(ranking_features)

        return scores.view(batch_size, num_stocks)

    def forward_features(self, src):
        """返回 (scores, features) 供对抗学习使用"""
        batch_size, num_stocks, seq_len, feature_dim = src.shape
        x = src.view(batch_size * num_stocks, seq_len, feature_dim)

        if self.industry_embed_dim > 0:
            tech = x[..., :-14]
            ind = x[..., -14:]
            x = self.input_proj(tech)
            i = self.industry_compressor(ind)
            x = torch.cat([x, i], dim=-1)
            x = self.industry_fusion(x)
        else:
            x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.temporal_layers:
            x = layer(x)
        features = self.feature_attention(x)  # [B*N, d_model]

        stock_features = features.view(batch_size, num_stocks, -1)
        if self.stock_embed_dim > 0:
            stock_ids = torch.arange(num_stocks, device=src.device)
            stock_emb = self.stock_embedding(stock_ids)
            stock_emb = stock_emb.unsqueeze(0).expand(batch_size, -1, -1)
            stock_features = stock_features + self.stock_emb_proj(stock_emb)
        stock_features = self.cross_stock_attention(stock_features)
        stock_features = stock_features.view(batch_size * num_stocks, -1)

        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)

        return scores.view(batch_size, num_stocks), features

    def predict_with_mc_dropout(self, src, num_samples=20):
        """
        MC Dropout 推理: 多次前向传播取平均（亮度平均分配）
        在推理时保持 dropout 激活，多次采样后平均
        """
        self.train()  # 保持 dropout 激活！
        all_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                scores = self.forward(src)
                all_scores.append(scores)
        # 亮度平均
        avg_scores = torch.stack(all_scores).mean(dim=0)
        return avg_scores


# ============================================================
# 卷积专家 (TCN风格时序卷积)
# ============================================================
class CausalConv1d(nn.Module):
    """因果卷积: 只看过去，不看未来"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        # x: [B*N, seq_len, features] -> [B*N, features, seq_len]
        x = x.transpose(1, 2)
        out = self.conv(x)
        # 去掉未来信息（右侧padding）
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        return out.transpose(1, 2)  # [B*N, seq_len, out_channels]


class TemporalConvBlock(nn.Module):
    """TCN残差块 + 随机深度"""
    def __init__(self, channels, kernel_size, dilation, dropout=0.1, sd_prob=0.9):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.sd_prob = sd_prob
        # 1x1 conv for residual if shapes differ (they don't here, but kept for safety)
        self.downsample = nn.Identity()

    def forward(self, x):
        if self.training and torch.rand(1).item() > self.sd_prob:
            return x  # 随机灭灯

        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = out + residual
        if not self.training:
            out = out * self.sd_prob
        return out


class ConvStockExpert(nn.Module):
    """
    卷积专家: 用TCN风格膨胀卷积处理时序
    不同膨胀率捕获不同时间尺度模式
    """
    def __init__(self, input_dim, expert_config, num_stocks):
        super().__init__()
        cfg = expert_config
        hidden = cfg.get('hidden_channels', 256)
        self.d_model = hidden
        self.num_stocks = num_stocks
        self.cfg_name = cfg.get('name', 'conv_expert')
        self.mc_dropout_rate = cfg.get('mc_dropout_rate', 0.1)
        self.industry_embed_dim = cfg.get('industry_embed_dim', 0)
        self.stock_embed_dim = cfg.get('stock_embed_dim', 0)

        # 输入投影
        if self.industry_embed_dim > 0:
            tech_dim = input_dim - 14
            self.input_proj = nn.Linear(tech_dim, hidden)
            self.industry_compressor = nn.Linear(14, self.industry_embed_dim)
            self.industry_fusion = nn.Linear(hidden + self.industry_embed_dim, hidden)
        else:
            self.input_proj = nn.Linear(input_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.input_dropout = nn.Dropout(cfg['dropout'])

        # 个股Embedding
        if self.stock_embed_dim > 0:
            self.stock_embedding = nn.Embedding(num_stocks, self.stock_embed_dim)
            self.stock_emb_proj = nn.Linear(self.stock_embed_dim, hidden)

        # 多尺度TCN块: 不同膨胀率捕获短期/中期/长期模式
        self.tcn_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8, 16, 1, 2, 4]  # 多尺度膨胀
        for d in dilations:
            self.tcn_blocks.append(
                TemporalConvBlock(hidden, kernel_size=3, dilation=d,
                                  dropout=cfg['dropout'],
                                  sd_prob=cfg.get('sd_prob', 0.9))
            )

        # 特征聚合
        self.feature_attention = FeatureAttention(hidden, cfg['dropout'])

        # 股票间交互
        self.cross_stock_attention = CrossStockAttention(
            hidden, cfg.get('nhead', 4), cfg['dropout']
        )

        # 排序层
        self.ranking_layers = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
        )

        # 分数头
        self.score_head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(cfg['dropout'] * 0.5),
            nn.Linear(hidden // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.shape
        x = src.view(batch_size * num_stocks, seq_len, feature_dim)

        # 输入投影
        if self.industry_embed_dim > 0:
            tech = x[..., :-14]
            ind = x[..., -14:]
            x = self.input_proj(tech)
            i = self.industry_compressor(ind)
            x = torch.cat([x, i], dim=-1)
            x = self.industry_fusion(x)
        else:
            x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # 多尺度TCN处理
        for block in self.tcn_blocks:
            x = block(x)

        # 特征注意力聚合
        x = self.feature_attention(x)  # [B*N, hidden]

        # 股票间交互
        stock_features = x.view(batch_size, num_stocks, -1)
        if self.stock_embed_dim > 0:
            stock_ids = torch.arange(num_stocks, device=src.device)
            stock_emb = self.stock_embedding(stock_ids)
            stock_emb = stock_emb.unsqueeze(0).expand(batch_size, -1, -1)
            stock_features = stock_features + self.stock_emb_proj(stock_emb)
        stock_features = self.cross_stock_attention(stock_features)
        stock_features = stock_features.view(batch_size * num_stocks, -1)

        # 排序层
        ranking_features = self.ranking_layers(stock_features)

        # 分数头
        scores = self.score_head(ranking_features)
        return scores.view(batch_size, num_stocks)

    def forward_features(self, src):
        """返回 (scores, features) 供对抗学习使用"""
        batch_size, num_stocks, seq_len, feature_dim = src.shape
        x = src.view(batch_size * num_stocks, seq_len, feature_dim)

        if self.industry_embed_dim > 0:
            tech = x[..., :-14]
            ind = x[..., -14:]
            x = self.input_proj(tech)
            i = self.industry_compressor(ind)
            x = torch.cat([x, i], dim=-1)
            x = self.industry_fusion(x)
        else:
            x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        for block in self.tcn_blocks:
            x = block(x)
        features = self.feature_attention(x)  # [B*N, hidden]

        stock_features = features.view(batch_size, num_stocks, -1)
        if self.stock_embed_dim > 0:
            stock_ids = torch.arange(num_stocks, device=src.device)
            stock_emb = self.stock_embedding(stock_ids)
            stock_emb = stock_emb.unsqueeze(0).expand(batch_size, -1, -1)
            stock_features = stock_features + self.stock_emb_proj(stock_emb)
        stock_features = self.cross_stock_attention(stock_features)
        stock_features = stock_features.view(batch_size * num_stocks, -1)

        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)

        return scores.view(batch_size, num_stocks), features

    def predict_with_mc_dropout(self, src, num_samples=20):
        """MC Dropout 推理"""
        self.train()
        all_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                scores = self.forward(src)
                all_scores.append(scores)
        return torch.stack(all_scores).mean(dim=0)


# ============================================================
# 月份季节性专家 (Month Seasonality Expert)
# ============================================================
class MonthSeasonalExpert(nn.Module):
    """
    月份相关性专家: 专注于学习股票×月份的周期性模式
    - 每只股票在每个月份有不同的表现倾向（季报、行业周期等）
    - 用可学习的 Stock×Month Embedding 捕捉这种效应
    - 轻量级设计，作为集成中的"日历效应"专家
    """
    def __init__(self, input_dim, expert_config, num_stocks):
        super().__init__()
        cfg = expert_config
        self.d_model = cfg.get('d_model', 128)
        self.num_stocks = num_stocks
        self.cfg_name = cfg.get('name', 'month_seasonal')
        self.mc_dropout_rate = cfg.get('mc_dropout_rate', 0.1)

        # 月份embedding (1-12月)
        self.month_embedding = nn.Embedding(12, self.d_model // 2)

        # 股票embedding (捕捉每只股票的季节性偏好)
        self.stock_seasonal_embedding = nn.Embedding(num_stocks, self.d_model // 4)

        # 输入投影 (轻量)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
        )

        # 特征聚合: LSTM捕捉短期时序
        self.lstm = nn.LSTM(
            input_size=self.d_model // 2,
            hidden_size=self.d_model // 2,
            num_layers=1,
            batch_first=True,
            dropout=cfg.get('dropout', 0.1),
            bidirectional=True,
        )

        # 合并: LSTM输出(bidirectional=d_model) + 月份embedding(d_model//2) + 股票季节性(d_model//4)
        fusion_dim = self.d_model + self.d_model // 2 + self.d_model // 4

        self.seasonal_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.1)),
            nn.Linear(fusion_dim // 2, 1),
            nn.Softmax(dim=1),
        )

        # 排序层
        self.ranking_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.1)),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.1)),
        )

        self.score_head = nn.Sequential(
            nn.Linear(fusion_dim // 4, fusion_dim // 8),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.1) * 0.5),
            nn.Linear(fusion_dim // 8, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src, month_ids=None):
        """
        src: [batch, num_stocks, seq_len, feature_dim]
        month_ids: [batch] 每个样本对应的月份 (0-11), 可选
        """
        batch_size, num_stocks, seq_len, feature_dim = src.shape

        # 如果没有提供月份，从日期推算（此处用占位）
        if month_ids is None:
            # 默认使用均匀分布的月份（推理时可以从数据中提取）
            month_ids = torch.zeros(batch_size, dtype=torch.long, device=src.device)

        x = src.view(batch_size * num_stocks, seq_len, feature_dim)
        x = self.input_proj(x)  # [B*N, L, d_model//2]

        # LSTM时序编码
        lstm_out, _ = self.lstm(x)  # [B*N, L, d_model]
        lstm_features = lstm_out[:, -1, :]  # 取最后一个时间步 [B*N, d_model]

        # 月份embedding
        month_emb = self.month_embedding(month_ids)  # [B, d_model//2]
        month_emb = month_emb.unsqueeze(1).expand(-1, num_stocks, -1)  # [B, N, d_model//2]
        month_emb = month_emb.reshape(batch_size * num_stocks, -1)  # [B*N, d_model//2]

        # 股票季节性embedding
        stock_ids = torch.arange(num_stocks, device=src.device).unsqueeze(0).expand(batch_size, -1)
        stock_seasonal = self.stock_seasonal_embedding(stock_ids)  # [B, N, d_model//4]
        stock_seasonal = stock_seasonal.reshape(batch_size * num_stocks, -1)  # [B*N, d_model//4]

        # 融合所有特征
        fused = torch.cat([lstm_features, month_emb, stock_seasonal], dim=-1)  # [B*N, fusion_dim]

        # 季节性注意力聚合
        fused_expanded = fused.view(batch_size, num_stocks, -1)
        attn_weights = self.seasonal_attention(fused_expanded)
        seasonal_features = (fused_expanded * attn_weights).view(batch_size * num_stocks, -1)

        # 排序层
        ranking_features = self.ranking_layers(seasonal_features)
        scores = self.score_head(ranking_features)

        return scores.view(batch_size, num_stocks)

    def forward_features(self, src, month_ids=None):
        """返回 (scores, features) 供对抗学习使用"""
        scores = self.forward(src, month_ids)
        # 用 LSTM 最后隐状态作为特征（简化）
        batch_size, num_stocks = src.shape[0], src.shape[1]
        x = self.input_proj(src.view(batch_size * num_stocks, src.shape[2], src.shape[3]))
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :self.d_model // 2]
        return scores, features

    def predict_with_mc_dropout(self, src, num_samples=20):
        """MC Dropout 推理"""
        self.train()
        all_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                scores = self.forward(src)
                all_scores.append(scores)
        return torch.stack(all_scores).mean(dim=0)


# ============================================================
# 激进专家 (Aggressive Expert)
# ============================================================
class AggressiveExpert(nn.Module):
    """
    激进型专家: 高波动高回报风格
    - 更大模型容量 (wide & shallow)
    - 更低dropout → 敢于下注
    - 更关注尾部收益
    """
    def __init__(self, input_dim, expert_config, num_stocks):
        super().__init__()
        cfg = expert_config
        self.d_model = cfg.get('d_model', 512)
        self.num_stocks = num_stocks
        self.cfg_name = cfg.get('name', 'aggressive')
        self.mc_dropout_rate = cfg.get('mc_dropout_rate', 0.05)  # 很低

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.05)),
        )

        self.pos_encoder = PositionalEncoding(self.d_model, cfg.get('dropout', 0.05))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=cfg.get('nhead', 8),
            dim_feedforward=cfg.get('dim_feedforward', 1024),
            dropout=cfg.get('dropout', 0.05), batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.get('num_layers', 2))

        self.feature_attention = FeatureAttention(self.d_model, cfg.get('dropout', 0.05))
        self.cross_stock_attention = CrossStockAttention(self.d_model, cfg.get('nhead', 8), cfg.get('dropout', 0.05))

        # 激进排序层: 更宽，激活更陡
        self.ranking_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.LeakyReLU(0.1),  # 负区间也有梯度 → 更激进
            nn.Dropout(cfg.get('dropout', 0.05)),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LeakyReLU(0.1),
        )

        self.score_head = nn.Sequential(
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(self.d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.5)  # 更大的初始化 → 更激进
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src):
        B, N, L, F = src.shape
        x = self.input_proj(src.view(B*N, L, F))
        x = self.pos_encoder(x)
        x = self.temporal_encoder(x)
        x = self.feature_attention(x)
        x = x.view(B, N, -1)
        x = self.cross_stock_attention(x)
        x = x.view(B*N, -1)
        x = self.ranking_layers(x)
        return self.score_head(x).view(B, N)

    def forward_features(self, src):
        B, N, L, F = src.shape
        x = self.input_proj(src.view(B*N, L, F))
        x = self.pos_encoder(x)
        x = self.temporal_encoder(x)
        features = self.feature_attention(x)
        sf = features.view(B, N, -1)
        sf = self.cross_stock_attention(sf)
        sf = sf.view(B*N, -1)
        scores = self.score_head(self.ranking_layers(sf))
        return scores.view(B, N), features

    def predict_with_mc_dropout(self, src, num_samples=20):
        self.train()
        all_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                all_scores.append(self.forward(src))
        return torch.stack(all_scores).mean(dim=0)


# ============================================================
# 布朗运动疯子专家 (Brownian Motion Noise Expert)
# ============================================================
class BrownianNoiseExpert(nn.Module):
    """
    布朗运动噪声专家: 训练时注入随机噪声模拟市场混沌
    - 前向传播时给输入加高斯噪声
    - 噪声强度随时间步增长（模拟布朗运动扩散）
    - 迫使模型在极端噪声下仍能排序 → 抗扰动能力强
    - 偶尔会有疯狂但正确的预测
    """
    def __init__(self, input_dim, expert_config, num_stocks):
        super().__init__()
        cfg = expert_config
        self.d_model = cfg.get('d_model', 256)
        self.num_stocks = num_stocks
        self.cfg_name = cfg.get('name', 'brownian')
        self.noise_base = cfg.get('noise_base', 0.02)
        self.noise_max = cfg.get('noise_max', 0.15)
        self.mc_dropout_rate = cfg.get('mc_dropout_rate', 0.12)

        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, cfg.get('dropout', 0.15))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=cfg.get('nhead', 4),
            dim_feedforward=cfg.get('dim_feedforward', 512),
            dropout=cfg.get('dropout', 0.15), batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.get('num_layers', 3))

        self.feature_attention = FeatureAttention(self.d_model, cfg.get('dropout', 0.15))
        self.cross_stock_attention = CrossStockAttention(self.d_model, cfg.get('nhead', 4), cfg.get('dropout', 0.15))

        self.ranking_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.15)),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
        )

        self.score_head = nn.Sequential(
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _brownian_noise(self, x, epoch_progress=0.5):
        """
        生成布朗运动风格噪声
        - 噪声强度随epoch进度增加（越训越疯）
        - 序列维度上累加 → 模拟随机游走
        """
        noise_level = self.noise_base + (self.noise_max - self.noise_base) * epoch_progress

        # 高斯噪声基底
        gaussian = torch.randn_like(x) * noise_level

        # 沿时间维累加 → 布朗运动
        brownian = torch.cumsum(gaussian, dim=-2) * 0.1

        return gaussian + brownian

    def forward(self, src, epoch_progress=0.5, add_noise=None):
        B, N, L, F = src.shape
        x = src.view(B*N, L, F)

        # 训练时注入布朗噪声
        if add_noise is None:
            add_noise = self.training

        if add_noise:
            x = x + self._brownian_noise(x, epoch_progress)

        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.temporal_encoder(x)
        features = self.feature_attention(x)
        sf = features.view(B, N, -1)
        sf = self.cross_stock_attention(sf)
        sf = sf.view(B*N, -1)
        scores = self.score_head(self.ranking_layers(sf))
        return scores.view(B, N)

    def forward_features(self, src):
        return self.forward(src, add_noise=False), None

    def predict_with_mc_dropout(self, src, num_samples=20):
        """MC Dropout + 微小噪声推理"""
        self.train()
        all_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                # 推理时加微小噪声（布朗运动风格）
                scores = self.forward(src, epoch_progress=0.3, add_noise=True)
                all_scores.append(scores)
        return torch.stack(all_scores).mean(dim=0)


# ============================================================
# 统计套利回归专家 (Statistical Arbitrage Regression)
# ============================================================
class NeuralGARCH(nn.Module):
    """
    神经GARCH波动率模型:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·h_t
    其中 h_t 是神经网络提取的隐藏状态
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # GARCH基参数（可学习）
        self.omega = nn.Parameter(torch.tensor(0.01))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))

        # 神经增强：从隐藏状态学习额外的波动率成分
        self.neural_vol = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # 保证正数
        )

    def forward(self, returns, hidden_state):
        """
        returns: [B*N, L] 收益率序列
        hidden_state: [B*N, L, d_model] 时序特征
        返回: [B*N, L] 条件波动率
        """
        BxN, L = returns.shape
        sigma2_list = [returns.var(dim=1)]  # 初始方差

        for t in range(1, L):
            epsilon2 = returns[:, t-1] ** 2
            neural_comp = self.neural_vol(hidden_state[:, t-1, :]).squeeze(-1)
            new_sigma = (
                torch.abs(self.omega) +
                torch.abs(self.alpha) * epsilon2 +
                torch.abs(self.beta) * sigma2_list[-1] +
                0.1 * neural_comp
            )
            sigma2_list.append(new_sigma)

        sigma2 = torch.stack(sigma2_list, dim=1)  # [B*N, L]
        return torch.sqrt(sigma2 + 1e-8)


class StatArbRegressionExpert(nn.Module):
    """
    GARCH+统计套利回归专家:
    - Neural GARCH 估计时变波动率
    - 波动率调整后的残差 → 识别真正的均值回复机会
    - 高波动+偏离大 → 可能是恐慌，即将反弹
    - 低波动+偏离小 → 趋势延续
    """
    def __init__(self, input_dim, expert_config, num_stocks):
        super().__init__()
        cfg = expert_config
        self.d_model = cfg.get('d_model', 192)
        self.num_stocks = num_stocks
        self.cfg_name = cfg.get('name', 'statarb')
        self.mc_dropout_rate = cfg.get('mc_dropout_rate', 0.1)
        self.lookback = cfg.get('statarb_lookback', 20)

        # 时序特征提取
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, cfg.get('dropout', 0.1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=cfg.get('nhead', 4),
            dim_feedforward=cfg.get('dim_feedforward', 384),
            dropout=cfg.get('dropout', 0.1), batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.get('num_layers', 2))

        # GARCH波动率模块
        self.garch = NeuralGARCH(self.d_model, cfg.get('dropout', 0.1))

        # 截面残差计算模块 (加入波动率)
        # forward中拼接: x_encoded(192) + zscore(1) + vol_regime(1) + current_vol(1) = 195
        self.residual_proj = nn.Sequential(
            nn.Linear(self.d_model + 3, self.d_model),  # +3: zscore, vol_regime, current_vol
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.1)),
        )

        # 均值回复预测头
        self.reversion_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.1)),
            nn.Linear(self.d_model // 2, 1),
            nn.Tanh(),
        )

        # 排序层
        self.ranking_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.1)),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
        )

        self.score_head = nn.Sequential(
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_cross_sectional_residuals(self, x, returns):
        """
        计算截面残差:
        - x: [B*N, L, d_model] 时序特征
        - returns: [B*N, L] 收益率序列
        返回每只股票相对横截面的偏离程度
        """
        BxN, L, D = x.shape
        # 取最近lookback天的特征
        recent = x[:, -self.lookback:, :]  # [BxN, lookback, D]
        recent_ret = returns[:, -self.lookback:]  # [BxN, lookback]

        # 截面均值（在股票维度）
        # 需要reshape: [B, N, lookback, D]
        # 简化：直接计算每个股票的统计量
        feat_mean = recent.mean(dim=1)  # [BxN, D]

        # 计算在特征空间中的偏离（相对其他股票）
        # 这里用batch内归一化近似截面
        feat_std = recent.std(dim=1) + 1e-8

        # 残差 = 当前值偏离均值的程度
        current = x[:, -1, :self.d_model]  # [BxN, d_model]
        residual = (current - feat_mean[:, :self.d_model]) / feat_std[:, :self.d_model]

        # 收益率在横截面中的排名
        ret_recent_avg = recent_ret.mean(dim=1)  # [BxN]
        # batch内排名近似
        spread_rank = torch.zeros_like(ret_recent_avg)
        ret_std = recent_ret.std(dim=1) + 1e-8
        zscore = (ret_recent_avg - ret_recent_avg.mean()) / (ret_recent_avg.std() + 1e-8)

        return residual, spread_rank, zscore

    def forward(self, src):
        B, N, L, F = src.shape
        x = src.view(B*N, L, F)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # 提取收益率（用收盘价变化近似）
        close_prices = src.view(B*N, L, F)[:, :, 2]  # 收盘价 (feature index 2)
        returns = (close_prices[:, 1:] - close_prices[:, :-1]) / (close_prices[:, :-1] + 1e-8)
        returns = torch.nn.functional.pad(returns, (1, 0))

        # 时序编码
        x_encoded = self.temporal_encoder(x)

        # GARCH波动率估计
        volatility = self.garch(returns, x_encoded)  # [B*N, L]

        # 截面残差（波动率调整后）
        current_vol = volatility[:, -1]  # [B*N]
        current_ret = returns[:, -1]  # [B*N]
        vol_adj_ret = current_ret / (current_vol + 1e-8)  # 波动率调整收益

        # 横截面zscore（波动率调整后更有意义）
        vol_adj_mean = vol_adj_ret.mean()
        vol_adj_std = vol_adj_ret.std() + 1e-8
        zscore = (vol_adj_ret - vol_adj_mean) / vol_adj_std

        # 计算偏离度
        short_term_vol = volatility[:, -5:].mean(dim=1)
        long_term_vol = volatility[:, -20:].mean(dim=1)
        vol_regime = short_term_vol / (long_term_vol + 1e-8)  # >1 = 高波动状态

        # 拼接特征
        x_aug = torch.cat([
            x_encoded[:, -1, :],          # [B*N, d_model]
            zscore.unsqueeze(1),          # 截面偏离
            vol_regime.unsqueeze(1),      # 波动率状态
            current_vol.unsqueeze(1),     # 当前波动率
        ], dim=1)

        # 特征增强
        features = self.residual_proj(x_aug)

        # 均值回复信号 [-1, 1]
        reversion_signal = self.reversion_head(features)

        # 排序分数
        ranking_features = self.ranking_layers(features)
        base_scores = self.score_head(ranking_features)

        # GARCH增强：高波动+偏离大 → 可能是恐慌底 → 加分
        panic_bonus = reversion_signal * torch.abs(zscore.unsqueeze(1)) * (vol_regime.unsqueeze(1) - 1).clamp(min=0)
        scores = base_scores + 0.3 * panic_bonus

        return scores.view(B, N)

    def forward_features(self, src):
        scores = self.forward(src)
        return scores, None

    def predict_with_mc_dropout(self, src, num_samples=20):
        self.train()
        all_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                all_scores.append(self.forward(src))
        return torch.stack(all_scores).mean(dim=0)


# ============================================================
# 对抗学习组件 (Gradient Reversal + 时间域分类器)
# ============================================================
class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层: 前向不变，反向时梯度取反并缩放"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)


class TimeDomainDiscriminator(nn.Module):
    """
    时间域判别器: 判断特征来自哪个时间段（月份/季度）
    通过对抗训练迫使特征提取器学习时间不变表示
    """
    def __init__(self, d_model, num_time_domains=12, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_time_domains),
        )

    def forward(self, features):
        # features: [B*N, d_model]
        return self.classifier(features)


class AdversarialStockExpert(nn.Module):
    """
    带对抗学习的股票专家:
    - 主任务: 排序选股
    - 对抗任务: 时间域分类（迫使特征时间不变性）
    将 Transformer 或 Conv 专家包装在对抗训练框架中
    """
    def __init__(self, base_expert, d_model, num_time_domains=12, adv_lambda=0.1):
        super().__init__()
        self.base_expert = base_expert
        self.d_model = d_model
        self.grl = GRL(lambda_=adv_lambda)
        self.domain_discriminator = TimeDomainDiscriminator(
            d_model, num_time_domains, dropout=0.1
        )
        self.adv_lambda = adv_lambda

    def forward(self, src, return_features=False):
        # 通过基础专家的 forward_features 提取特征
        scores, features = self.base_expert.forward_features(src)
        if return_features:
            return scores, features
        return scores

    def get_adversarial_loss(self, features, time_labels):
        """计算对抗损失: 判别器尝试预测时间域，但特征提取器要最大化判别器错误"""
        reversed_features = self.grl(features)
        domain_preds = self.domain_discriminator(reversed_features)
        return F.cross_entropy(domain_preds, time_labels)

    def predict_with_mc_dropout(self, src, num_samples=20):
        self.train()
        all_scores = []
        with torch.no_grad():
            for _ in range(num_samples):
                scores, _ = self.forward(src, return_features=True)
                all_scores.append(scores)
        return torch.stack(all_scores).mean(dim=0)


# ============================================================
# 元调度器 (Meta Aggregator)
# ============================================================
class MetaAggregator(nn.Module):
    """
    元调度器：学习如何组合多个专家的预测
    输入: [batch, num_stocks, num_experts] 各专家分数
    输出: [batch, num_stocks] 最终排序分数
    """
    def __init__(self, num_experts, num_stocks, hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts

        # 用注意力机制学习专家权重
        self.expert_attention = nn.Sequential(
            nn.Linear(num_experts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1),
        )

        # 股票间交互（在专家评分空间）
        # 确保 num_heads 能被 num_experts 整除
        valid_heads = [h for h in range(min(num_experts, 4), 0, -1) if num_experts % h == 0]
        nhead = valid_heads[0] if valid_heads else 1
        self.cross_stock_norm = nn.LayerNorm(num_experts)
        self.cross_stock_attn = nn.MultiheadAttention(
            num_experts, num_heads=nhead, dropout=0.1, batch_first=True
        )

        # 最终聚合
        self.aggregator = nn.Sequential(
            nn.Linear(num_experts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, expert_scores):
        # expert_scores: [batch, num_stocks, num_experts]
        batch_size, num_stocks, num_experts = expert_scores.shape

        # 1. 学习专家注意力权重
        attn_weights = self.expert_attention(expert_scores)  # [B, N, E]
        weighted_scores = expert_scores * attn_weights

        # 2. 股票间交互
        x = self.cross_stock_norm(weighted_scores)
        x, _ = self.cross_stock_attn(x, x, x)
        x = x + weighted_scores  # 残差

        # 3. 最终聚合为单分数
        final_scores = self.aggregator(x).squeeze(-1)  # [B, N]

        return final_scores


# ============================================================
# 集成预测器
# ============================================================
class EnsemblePredictor(nn.Module):
    """
    完整集成: 多个专家 + 元调度器
    """
    def __init__(self, experts, meta_aggregator, input_dim, expert_configs, num_stocks):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.meta = meta_aggregator
        self.num_experts = len(experts)

    def forward(self, src, mc_samples=20):
        """
        集成前向传播:
        1. 每个专家用 MC Dropout 生成预测（亮度平均）
        2. 元调度器融合各专家预测
        """
        batch_size, num_stocks = src.shape[0], src.shape[1]
        expert_scores = []

        for expert in self.experts:
            scores = expert.predict_with_mc_dropout(src, num_samples=mc_samples)
            expert_scores.append(scores)

        # 堆叠: [batch, num_stocks, num_experts]
        all_expert_scores = torch.stack(expert_scores, dim=-1)

        # 元调度器融合
        final_scores = self.meta(all_expert_scores)
        return final_scores, all_expert_scores
