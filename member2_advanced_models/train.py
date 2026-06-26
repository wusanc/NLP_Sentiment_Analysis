"""
成员2 - Attention-LSTM和Transformer训练脚本
职责：训练并评估Attention-LSTM和Transformer模型，保存结果
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
from member2_advanced_models.models import AttentionLSTM, TransformerClassifier
from member1_data_and_basic_models.word2vec import load_embedding_matrix

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
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
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估", leave=False):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(all_labels)
    metrics = evaluate(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


def train_model(model, model_name, train_loader, test_loader, device):
    print(f"\n{'='*50}")
    print(f"开始训练: {model_name}")
    print(f"{'='*50}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    best_f1 = 0
    for epoch in range(EPOCHS):
        start = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, metrics, preds, labels = evaluate_model(model, test_loader, criterion, device)
        scheduler.step()
        train_losses.append(t_loss); test_losses.append(v_loss)
        train_accs.append(t_acc); test_accs.append(metrics["accuracy"])
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS} [{elapsed:.1f}s] "
              f"Train Loss={t_loss:.4f} Acc={t_acc:.4f} | "
              f"Test Loss={v_loss:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            ckpt = os.path.join(BASE_DIR, "checkpoints", f"{model_name}.pt")
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(model.state_dict(), ckpt)
    print_report(labels, preds, model_name)
    save_result(model_name, metrics, train_losses, test_losses, train_accs, test_accs)
    return metrics


def run():
    print("=" * 60)
    print("成员2 - Attention-LSTM + Transformer训练")
    print("=" * 60)
    vocab = load_vocab()
    embedding_matrix = load_embedding_matrix()
    train_loader, test_loader, _ = get_dataloaders(vocab, batch_size=BATCH_SIZE)

    # Attention-LSTM
    attn_model = AttentionLSTM(
        vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
        pretrained_embeddings=embedding_matrix
    )
    attn_metrics = train_model(attn_model, "Attention-LSTM", train_loader, test_loader, DEVICE)

    # Transformer
    tf_model = TransformerClassifier(
        vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
        pretrained_embeddings=embedding_matrix
    )
    tf_metrics = train_model(tf_model, "Transformer", train_loader, test_loader, DEVICE)

    print(f"\n成员2任务完成！")
    print(f"  Attention-LSTM - Accuracy: {attn_metrics['accuracy']}, F1: {attn_metrics['f1']}")
    print(f"  Transformer    - Accuracy: {tf_metrics['accuracy']}, F1: {tf_metrics['f1']}")


if __name__ == "__main__":
    run()
