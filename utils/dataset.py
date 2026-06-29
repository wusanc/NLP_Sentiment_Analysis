"""
共用数据集工具
- 加载ChnSentiCorp数据
- 文本分词与词表构建
- PyTorch Dataset封装
"""

import os
import json
import pickle
import numpy as np
import jieba
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import List, Tuple, Dict, Optional


# ========== 配置 ==========
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 256
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def tokenize(text: str) -> List[str]:
    """使用jieba进行中文分词"""
    return list(jieba.cut(text.strip()))


def build_vocab(texts: List[List[str]], max_size: int = MAX_VOCAB_SIZE) -> Dict[str, int]:
    """根据分词结果构建词表"""
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab


def save_vocab(vocab: Dict[str, int], path: str = VOCAB_PATH):
    """保存词表到文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[词表] 已保存至 {path}，共 {len(vocab)} 个词", flush=True)


def load_vocab(path: str = VOCAB_PATH) -> Dict[str, int]:
    """加载词表"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_text(tokens: List[str], vocab: Dict[str, int], max_len: int = MAX_SEQ_LEN) -> List[int]:
    """将分词结果转为ID序列并padding"""
    unk_id = vocab.get(UNK_TOKEN, 1)
    ids = [vocab.get(t, unk_id) for t in tokens[:max_len]]
    # padding
    ids = ids + [0] * (max_len - len(ids))
    return ids


class SentimentDataset(Dataset):
    """情感分析PyTorch Dataset（直接使用预处理好的分词结果）"""

    def __init__(self, tokens: List[List[str]], labels: List[int], vocab: Dict[str, int],
                 max_len: int = MAX_SEQ_LEN):
        self.tokens = tokens
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        ids = encode_text(self.tokens[idx], self.vocab, self.max_len)
        length = min(len(self.tokens[idx]), self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_chnsenticorp(split: str = "train") -> Tuple[List[str], List[int]]:
    """
    加载ChnSentiCorp数据集（纯本地加载，不联网）
    返回: (texts, labels)  labels为0/1/2三分类
    """
    local_path = os.path.join(DATA_DIR, f"{split}.csv")
    if os.path.exists(local_path):
        import pandas as pd
        df = pd.read_csv(local_path, engine="python")
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        print(f"[数据集] 加载 {split} 集: {len(texts)} 条", flush=True)
        return texts, labels

    raise FileNotFoundError(
        f"无法加载数据集。请确保 data/{split}.csv 存在\n"
        f"可先运行 data/process_data.py 生成训练/测试集"
    )


def get_dataloaders(vocab: Optional[Dict[str, int]] = None, batch_size: int = 64,
                    max_len: int = MAX_SEQ_LEN):
    """
    获取训练集和测试集的DataLoader
    直接使用预处理好的分词结果，避免重复分词
    """
    # 加载预处理好的分词结果
    processed_dir = os.path.join(DATA_DIR, "processed")
    train_path = os.path.join(processed_dir, "train_tokens.pkl")
    test_path = os.path.join(processed_dir, "test_tokens.pkl")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "预处理分词结果不存在！请先运行 member1_data_and_basic_models.preprocess 进行数据预处理"
        )
    
    with open(train_path, "rb") as f:
        train_tokens, train_labels = pickle.load(f)
    with open(test_path, "rb") as f:
        test_tokens, test_labels = pickle.load(f)
    
    print(f"[数据集] 加载预处理分词结果: 训练集{len(train_tokens)}条, 测试集{len(test_tokens)}条", flush=True)

    # 构建或加载词表
    if vocab is None:
        if os.path.exists(VOCAB_PATH):
            vocab = load_vocab()
        else:
            vocab = build_vocab(train_tokens)
            save_vocab(vocab)
    print(f"[词表] 大小: {len(vocab)}", flush=True)

    # 创建Dataset
    train_dataset = SentimentDataset(train_tokens, train_labels, vocab, max_len)
    test_dataset = SentimentDataset(test_tokens, test_labels, vocab, max_len)

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader, vocab
