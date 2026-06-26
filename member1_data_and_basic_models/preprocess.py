"""
成员1 - 数据预处理模块
职责：文本清洗、jieba分词、构建词表、生成训练/测试数据
"""

import os
import sys
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.dataset import (
    tokenize, build_vocab, save_vocab, load_vocab,
    load_chnsenticorp, SentimentDataset, MAX_SEQ_LEN, VOCAB_PATH
)
import torch


def clean_text(text: str) -> str:
    """文本清洗：去除特殊字符、多余空格等"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）《》]', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()


def preprocess_data():
    """
    完整的数据预处理流程：
    1. 加载原始数据
    2. 文本清洗
    3. jieba分词
    4. 构建词表
    5. 保存处理后的数据
    """
    print("=" * 50)
    print("成员1 - 数据预处理")
    print("=" * 50)

    # 加载数据
    train_texts, train_labels = load_chnsenticorp("train")
    test_texts, test_labels = load_chnsenticorp("test")

    # 文本清洗
    print("\n[1/4] 文本清洗...")
    train_texts = [clean_text(t) for t in train_texts]
    test_texts = [clean_text(t) for t in test_texts]

    # 分词
    print("[2/4] jieba分词...")
    train_tokens = [tokenize(t) for t in train_texts]
    test_tokens = [tokenize(t) for t in test_texts]

    # 查看分词示例
    print("\n分词示例：")
    for i in range(3):
        print(f"  原文: {train_texts[i][:50]}...")
        print(f"  分词: {' / '.join(train_tokens[i][:10])}...")
        print()

    # 构建词表
    print("[3/4] 构建词表...")
    vocab = build_vocab(train_tokens)
    save_vocab(vocab)

    # 统计信息
    print("\n[4/4] 数据统计：")
    print(f"  训练集大小: {len(train_texts)}")
    print(f"  测试集大小: {len(test_texts)}")
    print(f"  词表大小: {len(vocab)}")
    print(f"  最大序列长度: {MAX_SEQ_LEN}")

    # 统计标签分布
    from collections import Counter
    train_dist = Counter(train_labels)
    print(f"  训练集标签分布: 正面={train_dist[1]}, 负面={train_dist[0]}")
    test_dist = Counter(test_labels)
    print(f"  测试集标签分布: 正面={test_dist[1]}, 负面={test_dist[0]}")

    # 统计句子长度分布
    lengths = [len(t) for t in train_tokens]
    print(f"  平均句子长度: {sum(lengths)/len(lengths):.1f} 词")
    print(f"  最大句子长度: {max(lengths)} 词")
    print(f"  中位数句子长度: {sorted(lengths)[len(lengths)//2]} 词")

    print("\n数据预处理完成！")
    return vocab, train_tokens, test_tokens


if __name__ == "__main__":
    preprocess_data()
