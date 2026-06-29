"""
成员3 - 可视化模块
职责：生成全部可视化图表，包括：
  - 五模型准确率/指标对比柱状图
  - 训练Loss曲线对比
  - 各模型混淆矩阵
  - 情感分布饼图
  - 词云图
  - Embedding降维可视化
"""

import os
import sys
import json
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.metrics import load_all_results, RESULTS_DIR

PLOT_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_COLORS = {
    "RNN": "#4CAF50",
    "LSTM": "#2196F3",
    "Attention-LSTM": "#FF9800",
    "Transformer": "#9C27B0",
    "BERT": "#F44336"
}


def plot_accuracy_comparison(results):
    """模型准确率对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["model_name"] for r in results]
    accs = [r["metrics"]["accuracy"] for r in results]
    colors = [MODEL_COLORS.get(n, "#607D8B") for n in names]
    bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_title("模型准确率对比", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[可视化] {path}")


def plot_f1_comparison(results):
    """模型F1分数对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["model_name"] for r in results]
    f1s = [r["metrics"]["f1"] for r in results]
    colors = [MODEL_COLORS.get(n, "#607D8B") for n in names]
    bars = ax.bar(names, f1s, color=colors, width=0.5, edgecolor="white")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{f1:.4f}", ha="center", fontsize=11)
    ax.set_ylabel("F1 Score")
    ax.set_title("模型F1分数对比", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "f1_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[可视化] {path}")


def plot_all_metrics(results):
    """全部指标综合对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    model_names = [r["model_name"] for r in results]
    x = np.arange(len(model_names))
    width = 0.18
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    for i, (m_name, label, color) in enumerate(zip(metrics_names, labels, colors)):
        vals = [r["metrics"][m_name] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Score")
    ax.set_title("五模型四指标综合对比", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "all_metrics_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[可视化] {path}")


def plot_training_loss(results):
    """训练Loss曲线对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        name = r["model_name"]
        color = MODEL_COLORS.get(name, "#607D8B")
        if r.get("train_losses"):
            axes[0].plot(r["train_losses"], label=name, color=color, linewidth=2)
        if r.get("test_losses"):
            axes[1].plot(r["test_losses"], label=name, color=color, linewidth=2)

    axes[0].set_title("训练集 Loss 曲线", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title("测试集 Loss 曲线", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "training_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[可视化] {path}")


def plot_accuracy_curve(results):
    """测试准确率曲线对比"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        name = r["model_name"]
        color = MODEL_COLORS.get(name, "#607D8B")
        if r.get("test_accs"):
            ax.plot(r["test_accs"], label=name, color=color, linewidth=2, marker="o", markersize=4)
    ax.set_title("测试集准确率变化曲线", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "accuracy_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[可视化] {path}")


def plot_sentiment_distribution():
    """数据集情感分布饼图"""
    from utils.dataset import load_chnsenticorp
    _, train_labels = load_chnsenticorp("train")
    counter = Counter(train_labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [counter.get(0, 0), counter.get(1, 0), counter.get(2, 0)]
    ax.pie(sizes, labels=["负面", "中性", "正面"], autopct="%1.1f%%",
           colors=["#F44336", "#FFC107", "#4CAF50"], startangle=90, textprops={"fontsize": 13})
    ax.set_title("ChnSentiCorp 训练集情感分布", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "sentiment_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[可视化] {path}")


def plot_wordcloud():
    """正面/负面评论词云图"""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[可视化] wordcloud未安装，跳过词云图")
        return

    from utils.dataset import load_chnsenticorp, tokenize
    train_texts, train_labels = load_chnsenticorp("train")

    pos_words = []
    neg_words = []
    neu_words = []
    stopwords = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
                 "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好"}

    for text, label in zip(train_texts[:2000], train_labels[:2000]):
        words = [w for w in tokenize(text) if w not in stopwords and len(w) > 1]
        if label == 2:
            pos_words.extend(words)
        elif label == 1:
            neu_words.extend(words)
        else:
            neg_words.extend(words)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    wc_pos = WordCloud(font_path="msyh.ttc", width=600, height=400,
                       background_color="white", max_words=100).generate(" ".join(pos_words))
    axes[0].imshow(wc_pos, interpolation="bilinear")
    axes[0].axis("off"); axes[0].set_title("正面评论词云", fontsize=13)

    wc_neg = WordCloud(font_path="msyh.ttc", width=600, height=400,
                       background_color="white", max_words=100).generate(" ".join(neg_words))
    axes[1].imshow(wc_neg, interpolation="bilinear")
    axes[1].axis("off"); axes[1].set_title("负面评论词云", fontsize=13)

    wc_neu = WordCloud(font_path="msyh.ttc", width=600, height=400,
                       background_color="white", max_words=100).generate(" ".join(neu_words))
    axes[2].imshow(wc_neu, interpolation="bilinear")
    axes[2].axis("off"); axes[2].set_title("中性评论词云", fontsize=13)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "wordcloud.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[可视化] {path}")


def generate_summary_table(results):
    """生成汇总表格"""
    print(f"\n{'='*70}")
    print(f"{'模型':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*70}")
    for r in results:
        m = r["metrics"]
        print(f"{r['model_name']:<18} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")
    print(f"{'='*70}")

    # 保存为文本文件
    path = os.path.join(PLOT_DIR, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'模型':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            m = r["metrics"]
            f.write(f"{r['model_name']:<18} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                    f"{m['recall']:>10.4f} {m['f1']:>10.4f}\n")
    print(f"[可视化] {path}")


def run():
    """运行全部可视化"""
    print("=" * 60)
    print("成员3 - 可视化生成")
    print("=" * 60)

    results = load_all_results()
    if not results:
        print("未找到训练结果，请先运行训练脚本")
        return

    print(f"找到 {len(results)} 个模型的结果: {[r['model_name'] for r in results]}")

    plot_accuracy_comparison(results)
    plot_f1_comparison(results)
    plot_all_metrics(results)
    plot_training_loss(results)
    plot_accuracy_curve(results)
    plot_sentiment_distribution()
    plot_wordcloud()
    generate_summary_table(results)

    print(f"\n所有图表已保存至 {PLOT_DIR}")


if __name__ == "__main__":
    run()
