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
    print(f"[数据] 词表已保存至 {path}，共 {len(vocab)} 个词")


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
    """情感分析PyTorch Dataset"""

    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int],
                 max_len: int = MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        ids = encode_text(tokens, self.vocab, self.max_len)
        length = min(len(tokens), self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_chnsenticorp(split: str = "train") -> Tuple[List[str], List[int]]:
    """
    加载ChnSentiCorp数据集
    优先从本地CSV加载，失败则从HuggingFace加载
    返回: (texts, labels)  labels为0/1二分类
    """
    # 优先从本地文件加载
    local_path = os.path.join(DATA_DIR, f"{split}.csv")
    if os.path.exists(local_path):
        import pandas as pd
        df = pd.read_csv(local_path)
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        # 标签映射：如果存在3类(0,1,2)，过滤掉中性(1)并将2映射为1
        unique_labels = set(labels)
        if unique_labels == {0, 1, 2}:
            filtered = [(t, l) for t, l in zip(texts, labels) if l != 1]
            texts = [t for t, l in filtered]
            labels = [l if l == 0 else 1 for _, l in filtered]
            print(f"[数据] 从本地文件加载 {split} 集: {len(texts)} 条 (已过滤中性)")
        else:
            print(f"[数据] 从本地文件加载 {split} 集: {len(texts)} 条")
        return texts, labels

    # 尝试从HuggingFace加载
    try:
        from datasets import load_dataset
        dataset = load_dataset("seamew/ChnSentiCorp", split=split)
        texts = dataset["text"]
        labels = dataset["label"]
        print(f"[数据] 从HuggingFace加载 {split} 集: {len(texts)} 条")
        return texts, labels
    except Exception as e:
        print(f"[数据] HuggingFace加载失败: {e}")

    raise FileNotFoundError(
        f"无法加载数据集。请确保 data/{split}.csv 存在"
    )


def get_dataloaders(vocab: Optional[Dict[str, int]] = None, batch_size: int = 64,
                    max_len: int = MAX_SEQ_LEN):
    """
    获取训练集和测试集的DataLoader
    如果未提供词表，则自动构建并保存
    """
    # 加载数据
    train_texts, train_labels = load_chnsenticorp("train")
    test_texts, test_labels = load_chnsenticorp("test")

    # 分词
    print("[数据] 正在分词...")
    train_tokens = [tokenize(t) for t in train_texts]

    # 构建或加载词表
    if vocab is None:
        if os.path.exists(VOCAB_PATH):
            vocab = load_vocab()
        else:
            vocab = build_vocab(train_tokens)
            save_vocab(vocab)
    print(f"[数据] 词表大小: {len(vocab)}")

    # 创建Dataset
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, max_len)

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader, vocab
