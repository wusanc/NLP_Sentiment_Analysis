"""
主入口 - 运行全部模型训练并生成可视化报告
用法: python main.py
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def check_data():
    """检查本地数据文件是否存在"""
    train_path = os.path.join(BASE_DIR, "data", "train.csv")
    test_path = os.path.join(BASE_DIR, "data", "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("错误：未找到数据文件！", flush=True)
        print("请先运行 data/process_data.py 生成训练/测试集", flush=True)
        sys.exit(1)
    print("[数据] 本地数据文件检查通过", flush=True)


def main():
    parser = argparse.ArgumentParser(description="中文情感分析系统")
    parser.add_argument("--member1", action="store_true", help="仅运行成员1(RNN+LSTM)")
    parser.add_argument("--member2", action="store_true", help="仅运行成员2(Attention+Transformer)")
    parser.add_argument("--member3", action="store_true", help="仅运行成员3(BERT)")
    parser.add_argument("--vis", action="store_true", help="仅运行可视化")
    args = parser.parse_args()

    run_all = not (args.member1 or args.member2 or args.member3 or args.vis)

    # ========================
    # 0. 数据检查
    # ========================
    print("\n" + "=" * 60, flush=True)
    print("[0/4] 数据检查", flush=True)
    print("=" * 60, flush=True)
    check_data()

    # ========================
    # 1. 成员1: 数据预处理 + Word2Vec + RNN/LSTM
    # ========================
    if run_all or args.member1:
        print("\n" + "=" * 60, flush=True)
        print("[1/4] 成员1 - 数据预处理 + Word2Vec + RNN/LSTM", flush=True)
        print("=" * 60, flush=True)
        from member1_data_and_basic_models.train import run as member1_run
        member1_run()

    # ========================
    # 2. 成员2: Attention-LSTM + Transformer
    # ========================
    if run_all or args.member2:
        print("\n" + "=" * 60, flush=True)
        print("[2/4] 成员2 - Attention-LSTM + Transformer", flush=True)
        print("=" * 60, flush=True)
        from member2_advanced_models.train import run as member2_run
        member2_run()

    # ========================
    # 3. 成员3: BERT微调
    # ========================
    if run_all or args.member3:
        print("\n" + "=" * 60, flush=True)
        print("[3/4] 成员3 - BERT微调", flush=True)
        print("=" * 60, flush=True)
        from member3_bert_and_visualization.train import run as member3_run
        member3_run()

    # ========================
    # 4. 可视化
    # ========================
    if run_all or args.vis:
        print("\n" + "=" * 60, flush=True)
        print("[4/4] 生成可视化报告", flush=True)
        print("=" * 60, flush=True)
        from member3_bert_and_visualization.visualization import run as vis_run
        vis_run()

    print("\n" + "=" * 60, flush=True)
    print("全部任务完成！", flush=True)
    print("=" * 60, flush=True)
    print("结果文件位于:", flush=True)
    print(f"  - 训练结果: {os.path.join(BASE_DIR, 'results')}/", flush=True)
    print(f"  - 模型权重: {os.path.join(BASE_DIR, 'checkpoints')}/", flush=True)
    print(f"  - 可视化图表: {os.path.join(BASE_DIR, 'results', 'plots')}/", flush=True)

if __name__ == "__main__":
    main()
