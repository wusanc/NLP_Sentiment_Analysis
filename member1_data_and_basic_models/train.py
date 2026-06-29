"""
成员1 - RNN和LSTM训练脚本
职责：训练并评估RNN和LSTM模型，保存结果供可视化使用
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.dataset import get_dataloaders, load_vocab
from utils.metrics import evaluate, print_report, save_result
from member1_data_and_basic_models.models import RNNClassifier, LSTMClassifier
from member1_data_and_basic_models.word2vec import load_embedding_matrix, train_word2vec

# 超参数
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="训练", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_model(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估", leave=False):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    metrics = evaluate(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


def train_model(model, model_name: str, train_loader, test_loader, device):
    """完整的训练流程"""
    print(f"\n{'='*50}", flush=True)
    print(f"训练模型: {model_name}", flush=True)
    print(f"{'='*50}", flush=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_f1 = 0

    for epoch in range(EPOCHS):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 评估
        test_loss, metrics, preds, labels = evaluate_model(model, test_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(metrics["accuracy"])

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:2d}/{EPOCHS} [{elapsed:5.1f}s] "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Test  Loss={test_loss:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}", flush=True)

        # 保存最佳模型
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            ckpt_path = os.path.join(BASE_DIR, "checkpoints", f"{model_name}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    # 打印最终报告
    print_report(labels, preds, model_name)

    # 保存结果
    save_result(model_name, metrics, train_losses, test_losses, train_accs, test_accs)

    print(f"\n{model_name} 训练完成！最佳F1: {best_f1:.4f}", flush=True)
    return metrics


def run():
    """运行成员1的所有任务"""
    print("=" * 60, flush=True)
    print("成员1 - 数据预处理 + Word2Vec + RNN/LSTM训练", flush=True)
    print("=" * 60, flush=True)

    # 1. 数据预处理
    from member1_data_and_basic_models.preprocess import preprocess_data
    preprocess_data()

    # 2. 训练Word2Vec
    w2v_model, embedding_matrix = train_word2vec(vector_size=EMBEDDING_DIM)

    # 3. 准备数据
    vocab = load_vocab()
    train_loader, test_loader, _ = get_dataloaders(vocab, batch_size=BATCH_SIZE)

    # 4. 训练RNN
    rnn_model = RNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        pretrained_embeddings=embedding_matrix
    )
    rnn_metrics = train_model(rnn_model, "RNN", train_loader, test_loader, DEVICE)

    # 5. 训练LSTM
    lstm_model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        pretrained_embeddings=embedding_matrix
    )
    lstm_metrics = train_model(lstm_model, "LSTM", train_loader, test_loader, DEVICE)

    print(f"\n{'='*60}", flush=True)
    print(f"成员1全部完成！", flush=True)
    print(f"  RNN  - Accuracy: {rnn_metrics['accuracy']:.4f}  F1: {rnn_metrics['f1']:.4f}", flush=True)
    print(f"  LSTM - Accuracy: {lstm_metrics['accuracy']:.4f}  F1: {lstm_metrics['f1']:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    run()
