# 中文情感分析系统 — 基于LSTM与Attention

## 项目概述

本项目实现一个中文情感分析系统，对文本进行正面/负面情感分类，并比较 RNN、LSTM、Attention-LSTM、Transformer 和 BERT 五种模型的效果。

## 团队分工（3人）

| 成员 | 负责模块 | 核心文件 |
|------|----------|----------|
| **成员1** | 数据预处理 + Word2Vec + RNN/LSTM | `member1_data_and_basic_models/` |
| **成员2** | Attention-LSTM + Transformer | `member2_advanced_models/` |
| **成员3** | BERT微调 + 全部可视化 + 主程序 | `member3_bert_and_visualization/` + `main.py` |

## 项目结构

```
NLP_Sentiment_Analysis/
├── data/
│   └── download_data.py          # 数据下载脚本
├── utils/
│   ├── dataset.py                 # 共用数据集类
│   └── metrics.py                 # 共用评估指标
├── member1_data_and_basic_models/
│   ├── preprocess.py              # 分词、词表构建
│   ├── word2vec.py                # Word2Vec训练
│   ├── models.py                  # RNN / LSTM 模型
│   └── train.py                   # 训练RNN和LSTM
├── member2_advanced_models/
│   ├── models.py                  # Attention-LSTM / Transformer 模型
│   └── train.py                   # 训练Attention-LSTM和Transformer
├── member3_bert_and_visualization/
│   ├── models.py                  # BERT微调模型
│   ├── train.py                   # 训练BERT
│   └── visualization.py           # 全部可视化（对比图、词云等）
├── checkpoints/                   # 模型权重保存
├── results/                       # 训练结果JSON
├── main.py                        # 主入口：运行全部模型+生成报告
├── requirements.txt
└── README.md
```

## 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据
python data/download_data.py

# 3. 运行全部（推荐）
python main.py

# 4. 或分别运行各成员代码
python -m member1_data_and_basic_models.train
python -m member2_advanced_models.train
python -m member3_bert_and_visualization.train
python -m member3_bert_and_visualization.visualization
```

## 数据集

使用 ChnSentiCorp 中文情感分析数据集（约 9600 条训练 / 1200 条测试），标签为正面(1)/负面(0)。
