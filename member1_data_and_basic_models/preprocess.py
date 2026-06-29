"""
成员1 - 数据预处理模块
职责：文本清洗、jieba分词、构建词表、生成训练/测试数据
"""

import os
import sys
import re
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.dataset import (
    tokenize, build_vocab, save_vocab, load_vocab,
    load_chnsenticorp, SentimentDataset, MAX_SEQ_LEN, VOCAB_PATH
)
import torch

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


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
    print("=" * 50, flush=True)
    print("成员1 - 数据预处理", flush=True)
    print("=" * 50, flush=True)

    # 加载数据
    print("\n[1/5] 加载数据...", flush=True)
    train_texts, train_labels = load_chnsenticorp("train")
    test_texts, test_labels = load_chnsenticorp("test")

    # 文本清洗
    print("\n[2/5] 文本清洗...", flush=True)
    train_texts = [clean_text(t) for t in train_texts]
    test_texts = [clean_text(t) for t in test_texts]
    print(f"  完成: 训练{len(train_texts)}条, 测试{len(test_texts)}条", flush=True)

    # 分词
    print("\n[3/5] jieba分词...", flush=True)
    train_tokens = [tokenize(t) for t in train_texts]
    test_tokens = [tokenize(t) for t in test_texts]
    print(f"  完成: 训练集{len(train_tokens)}条, 测试集{len(test_tokens)}条", flush=True)

    # 查看分词示例
    print("\n  分词示例：", flush=True)
    for i in range(3):
        print(f"    原文: {train_texts[i][:40]}...", flush=True)
        print(f"    分词: {' / '.join(train_tokens[i][:8])}...", flush=True)
        print()

    # 构建词表
    print("[4/5] 构建词表...", flush=True)
    vocab = build_vocab(train_tokens)
    save_vocab(vocab)

    # 统计信息
    print("\n[5/5] 数据统计：", flush=True)
    print(f"  训练集大小: {len(train_texts)}", flush=True)
    print(f"  测试集大小: {len(test_texts)}", flush=True)
    print(f"  词表大小: {len(vocab)}", flush=True)
    print(f"  最大序列长度: {MAX_SEQ_LEN}", flush=True)

    # 统计标签分布
    from collections import Counter
    train_dist = Counter(train_labels)
    print(f"  训练集标签: 负面={train_dist.get(0,0)}, 中性={train_dist.get(1,0)}, 正面={train_dist.get(2,0)}", flush=True)
    test_dist = Counter(test_labels)
    print(f"  测试集标签: 负面={test_dist.get(0,0)}, 中性={test_dist.get(1,0)}, 正面={test_dist.get(2,0)}", flush=True)

    # 统计句子长度分布
    lengths = [len(t) for t in train_tokens]
    print(f"  平均句长: {sum(lengths)/len(lengths):.1f} 词", flush=True)
    print(f"  最大句长: {max(lengths)} 词", flush=True)
    print(f"  中位句长: {sorted(lengths)[len(lengths)//2]} 词", flush=True)

    # 保存分词结果
    print("\n  保存分词结果...", flush=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "train_tokens.pkl"), "wb") as f:
        pickle.dump((train_tokens, train_labels), f)
    with open(os.path.join(PROCESSED_DIR, "test_tokens.pkl"), "wb") as f:
        pickle.dump((test_tokens, test_labels), f)
    print(f"  已保存至 {PROCESSED_DIR}", flush=True)

    print("\n数据预处理完成！", flush=True)
    return vocab, train_tokens, test_tokens


if __name__ == "__main__":
    preprocess_data()
