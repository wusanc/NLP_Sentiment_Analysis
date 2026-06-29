"""
成员3 - BERT微调模型定义
职责：使用HuggingFace transformers实现BERT文本分类
"""

import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel

# 本地BERT模型路径（供train.py导入使用）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERT_MODEL_PATH = os.path.join(BASE_DIR, "data", "bert-base-chinese")


class BertClassifier(nn.Module):
    """基于BERT的中文情感分类模型"""

    def __init__(self, model_name: str = "bert-base-chinese", num_classes: int = 3,
                 dropout: float = 0.3, freeze_bert: bool = False):
        super().__init__()
        try:
            self.bert = BertModel.from_pretrained(model_name)
        except Exception as e:
            print(f"[BERT] 下载模型失败: {e}", flush=True)
            print("[BERT] 尝试离线加载...", flush=True)
            self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 使用<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token的输出
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits


class BertSentimentDataset(torch.utils.data.Dataset):
    """BERT专用数据集（使用BERT tokenizer）"""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
