"""
成员2 - Attention-LSTM和Transformer模型定义
职责：实现带注意力机制的LSTM和Transformer编码器分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    """自注意力机制"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(lstm_output))  # (batch, seq_len, hidden_dim)
        attention_scores = self.context(energy).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attention_weights


class AttentionLSTM(nn.Module):
    """带注意力机制的BiLSTM文本分类模型"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int = 2, num_layers: int = 1, dropout: float = 0.3,
                 pretrained_embeddings=None):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        lstm_output, (hidden, cell) = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        # 注意力
        context, attn_weights = self.attention(lstm_output)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    """基于Transformer编码器的文本分类模型"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int = 2, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.3, max_len: int = 256,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True

        self.pos_encoding = PositionalEncoding(embedding_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, lengths):
        batch_size, seq_len = input_ids.shape

        # 创建padding mask (True表示需要被mask的位置)
        mask = (input_ids == 0)  # (batch, seq_len)

        # Embedding + 位置编码
        emb = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        emb = self.pos_encoding(emb)
        emb = self.dropout(emb)

        # Transformer编码
        output = self.transformer_encoder(emb, src_key_padding_mask=mask)

        # 使用[CLS]位置(第一个token)的输出进行分类
        # 或者使用mean pooling
        # 这里使用mean pooling（排除padding）
        mask_expanded = (~mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_output = (output * mask_expanded).sum(dim=1)  # (batch, hidden)
        lengths_float = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = sum_output / lengths_float  # (batch, hidden)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
