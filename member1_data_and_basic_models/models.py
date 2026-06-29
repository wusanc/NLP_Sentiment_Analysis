"""
成员1 - RNN和LSTM模型定义
职责：实现基础的RNN和LSTM文本分类模型
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNClassifier(nn.Module):
    """简单RNN文本分类模型"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int = 3, num_layers: int = 1, dropout: float = 0.3,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        hidden = hidden[-1]  # 取最后一层的隐藏状态
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


class LSTMClassifier(nn.Module):
    """LSTM文本分类模型"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int = 3, num_layers: int = 1, dropout: float = 0.3,
                 pretrained_embeddings=None):
        super().__init__()
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
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向，所以乘2

    def forward(self, input_ids, lengths):
        emb = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        # 拼接正向和反向的最后隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits
