"""
共用评估指标
- 计算准确率、精确率、召回率、F1
- 生成分类报告和混淆矩阵
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """计算所有评估指标"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4)
    }


def print_report(y_true: List[int], y_pred: List[int], model_name: str = ""):
    """打印详细分类报告"""
    print(f"\n{'='*50}", flush=True)
    print(f"模型: {model_name} - 测试集结果", flush=True)
    print(f"{'='*50}", flush=True)
    print(classification_report(y_true, y_pred, target_names=["负面", "中性", "正面"], zero_division=0), flush=True)
    cm = confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵:\n{cm}", flush=True)


def save_result(model_name: str, metrics: Dict, train_losses: List[float] = None,
                test_losses: List[float] = None, train_accs: List[float] = None,
                test_accs: List[float] = None):
    """保存训练结果到JSON文件，供可视化使用"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "model_name": model_name,
        "metrics": metrics,
        "train_losses": train_losses or [],
        "test_losses": test_losses or [],
        "train_accs": train_accs or [],
        "test_accs": test_accs or []
    }
    path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[结果] {model_name} 已保存至 {path}", flush=True)


def load_all_results() -> List[Dict]:
    """加载所有模型的训练结果"""
    results = []
    if not os.path.exists(RESULTS_DIR):
        return results
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(RESULTS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
    return results
