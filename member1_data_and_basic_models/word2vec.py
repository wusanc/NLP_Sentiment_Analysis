"""
成员1 - Word2Vec词向量训练模块
职责：使用gensim训练Word2Vec模型，生成词嵌入矩阵供后续模型使用
"""

import os
import sys
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gensim.models import Word2Vec
from utils.dataset import load_vocab, tokenize, load_chnsenticorp, VOCAB_PATH

W2V_DIR = os.path.join(BASE_DIR, "checkpoints")
W2V_PATH = os.path.join(W2V_DIR, "word2vec.model")
EMBEDDING_MATRIX_PATH = os.path.join(W2V_DIR, "embedding_matrix.pkl")


def train_word2vec(vector_size: int = 128, window: int = 5, min_count: int = 1,
                   epochs: int = 10):
    """
    训练Word2Vec模型

    参数:
        vector_size: 词向量维度
        window: 上下文窗口大小
        min_count: 最小词频
        epochs: 训练轮数
    """
    print("=" * 50)
    print("成员1 - Word2Vec词向量训练")
    print("=" * 50)

    # 加载并分词
    print("\n[1/3] 加载数据并分词...", flush=True)
    train_texts, _ = load_chnsenticorp("train")
    test_texts, _ = load_chnsenticorp("test")
    all_texts = train_texts + test_texts
    sentences = [tokenize(t) for t in all_texts]
    print(f"  总句子数: {len(sentences)}", flush=True)

    # 训练Word2Vec
    print(f"\n[2/3] 训练Word2Vec (dim={vector_size}, window={window})...", flush=True)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        epochs=epochs,
        sg=1
    )

    # 保存模型
    os.makedirs(W2V_DIR, exist_ok=True)
    model.save(W2V_PATH)
    print(f"  Word2Vec模型已保存至 {W2V_PATH}", flush=True)
    print(f"  词汇量: {len(model.wv)}", flush=True)

    # 生成embedding matrix
    print("\n[3/3] 生成Embedding矩阵...", flush=True)
    embedding_matrix = get_embedding_matrix(model, vector_size)

    # 测试词向量
    print("\n词向量相似词测试：", flush=True)
    test_words = ["不错", "好看", "差", "喜欢"]
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            similar_str = ", ".join([f"{w}({s:.2f})" for w, s in similar])
            print(f"  '{word}' 的相似词: {similar_str}")

    return model, embedding_matrix


def get_embedding_matrix(w2v_model=None, vector_size: int = 128, vocab=None):
    """
    根据Word2Vec模型和词表生成Embedding矩阵

    返回: numpy数组, shape=(vocab_size, vector_size)
    """
    if vocab is None:
        vocab = load_vocab()

    if w2v_model is None:
        if os.path.exists(W2V_PATH):
            w2v_model = Word2Vec.load(W2V_PATH)
        else:
            raise FileNotFoundError("Word2Vec模型未找到，请先运行训练")

    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, vector_size))

    found = 0
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            found += 1

    print(f"  Embedding矩阵: ({vocab_size}, {vector_size})", flush=True)
    print(f"  词向量覆盖率: {found}/{vocab_size} ({found/vocab_size*100:.1f}%)", flush=True)

    # 保存
    with open(EMBEDDING_MATRIX_PATH, "wb") as f:
        pickle.dump(embedding_matrix, f)
    print(f"  Embedding矩阵已保存至 {EMBEDDING_MATRIX_PATH}", flush=True)

    return embedding_matrix


def load_embedding_matrix():
    """加载预训练的Embedding矩阵"""
    if os.path.exists(EMBEDDING_MATRIX_PATH):
        with open(EMBEDDING_MATRIX_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return get_embedding_matrix()


if __name__ == "__main__":
    train_word2vec()
