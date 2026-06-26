"""
成员3 - BERT微调训练脚本
职责：训练并评估BERT模型，保存结果供可视化使用
"""

import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.dataset import load_chnsenticorp
from utils.metrics import evaluate, print_report, save_result
from member3_bert_and_visualization.models import BertClassifier, BertSentimentDataset
from transformers import BertTokenizer

MODEL_NAME = "bert-base-chinese"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bert_dataloaders(batch_size=32, max_len=128):
    """使用BERT tokenizer加载数据"""
    print("[BERT] 加载数据...")
    train_texts, train_labels = load_chnsenticorp("train")
    test_texts, test_labels = load_chnsenticorp("test")

    print("[BERT] 加载BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = BertSentimentDataset(train_texts, train_labels, tokenizer, max_len)
    test_dataset = BertSentimentDataset(test_texts, test_labels, tokenizer, max_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"[BERT] 训练集: {len(train_dataset)}条, 测试集: {len(test_dataset)}条")
    return train_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(dataloader, desc="训练", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

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
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(all_labels)
    metrics = evaluate(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


def run():
    print("=" * 60)
    print("成员3 - BERT微调训练")
    print("=" * 60)

    # 加载数据
    train_loader, test_loader = get_bert_dataloaders(BATCH_SIZE, MAX_LEN)

    # 创建模型
    print(f"\n[BERT] 加载预训练模型: {MODEL_NAME}")
    model = BertClassifier(model_name=MODEL_NAME)
    model = model.to(DEVICE)

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    best_f1 = 0

    for epoch in range(EPOCHS):
        start = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, metrics, preds, labels = evaluate_model(model, test_loader, criterion, DEVICE)
        elapsed = time.time() - start

        train_losses.append(t_loss); test_losses.append(v_loss)
        train_accs.append(t_acc); test_accs.append(metrics["accuracy"])

        print(f"Epoch {epoch+1}/{EPOCHS} [{elapsed:.1f}s] "
              f"Train Loss={t_loss:.4f} Acc={t_acc:.4f} | "
              f"Test Loss={v_loss:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            ckpt = os.path.join(BASE_DIR, "checkpoints", "BERT.pt")
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(model.state_dict(), ckpt)

    print_report(labels, preds, "BERT")
    save_result("BERT", metrics, train_losses, test_losses, train_accs, test_accs)
    print(f"\n成员3 BERT训练完成！Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}")


if __name__ == "__main__":
    run()
