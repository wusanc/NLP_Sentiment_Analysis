"""
主入口 - 运行全部模型训练并生成可视化报告
用法: python main.py
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def main():
    parser = argparse.ArgumentParser(description="中文情感分析系统 - 基于LSTM与Attention")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过数据预处理")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据下载")
    parser.add_argument("--member1", action="store_true", help="仅运行成员1(RNN+LSTM)")
    parser.add_argument("--member2", action="store_true", help="仅运行成员2(Attention+Transformer)")
    parser.add_argument("--member3", action="store_true", help="仅运行成员3(BERT)")
    parser.add_argument("--vis", action="store_true", help="仅运行可视化")
    args = parser.parse_args()

    run_all = not (args.member1 or args.member2 or args.member3 or args.vis)

    # ========================
    # 0. 数据下载
    # ========================
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("[0/4] 数据下载")
        print("=" * 60)
        train_path = os.path.join(BASE_DIR, "data", "train.csv")

    # ========================
    # 1. 成员1: 数据预处理 + Word2Vec + RNN/LSTM
    # ========================
    if run_all or args.member1:
        print("\n" + "=" * 60)
        print("[1/4] 成员1 - 数据预处理 + Word2Vec + RNN/LSTM")
        print("=" * 60)
        from member1_data_and_basic_models.train import run as member1_run
        member1_run()

    # ========================
    # 2. 成员2: Attention-LSTM + Transformer
    # ========================
    if run_all or args.member2:
        print("\n" + "=" * 60)
        print("[2/4] 成员2 - Attention-LSTM + Transformer")
        print("=" * 60)
        from member2_advanced_models.train import run as member2_run
        member2_run()

    # ========================
    # 3. 成员3: BERT微调
    # ========================
    if run_all or args.member3:
        print("\n" + "=" * 60)
        print("[3/4] 成员3 - BERT微调")
        print("=" * 60)
        from member3_bert_and_visualization.train import run as member3_run
        member3_run()

    # ========================
    # 4. 可视化
    # ========================
    if run_all or args.vis:
        print("\n" + "=" * 60)
        print("[4/4] 生成可视化报告")
        print("=" * 60)
        from member3_bert_and_visualization.visualization import run as vis_run
        vis_run()

    print("\n" + "=" * 60)
    print("全部任务完成！")
    print("=" * 60)
    print("结果文件位于:")
    print(f"  - 训练结果: {os.path.join(BASE_DIR, 'results')}/")
    print(f"  - 模型权重: {os.path.join(BASE_DIR, 'checkpoints')}/")
    print(f"  - 可视化图表: {os.path.join(BASE_DIR, 'results', 'plots')}/")


if __name__ == "__main__":
    main()
