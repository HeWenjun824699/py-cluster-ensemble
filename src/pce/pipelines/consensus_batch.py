import os
import glob
import time
import scipy.io
import numpy as np
from pathlib import Path

# 引入你的库组件
import pce.io
import pce.generators
import pce.consensus
import pce.metrics


def consensus_batch(
        input_dir: str,
        output_dir: str,
        consensus_method: str = 'cspa',
        generator_method: str = 'cdkmeans',
        n_base: int = 20,
        seed: int = 2024
):
    """
    批量执行聚类集成流水线。

    Args:
        input_dir: 输入数据集目录 (.mat)
        output_dir: 结果输出目录
        consensus_method: 'cspa', 'mcla', 'hgpa' 等 (函数名字符串)
        generator_method: 如果数据是原始 X，使用该生成器 (如 'cdkmeans', 'litekmeans')
        n_base: 基聚类器数量 (仅当需要生成 BPs 时使用)
        seed: 随机种子
    """
    # 1. 准备目录
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not output_path.exists():
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # 2. 获取算法函数 (利用 getattr 动态获取)
    try:
        consensus_func = getattr(pce.consensus, consensus_method)
    except AttributeError:
        raise ValueError(f"Consensus method '{consensus_method}' not found in pce.consensus")

    try:
        generator_func = getattr(pce.generators, generator_method)
    except AttributeError:
        raise ValueError(f"Generator method '{generator_method}' not found in pce.generators")

    # 3. 遍历文件
    mat_files = list(input_path.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return

    print(f"Found {len(mat_files)} datasets. Starting batch process with [{consensus_method}]...")

    for file_path in mat_files:
        dataset_name = file_path.stem  # 获取文件名（不含后缀）
        print(f"\n>>> Processing: {dataset_name}")

        try:
            # --- A. 数据加载与探测 ---
            # 为了判断包含 X 还是 BPs，我们先用 scipy 加载 raw dict 查看 keys
            raw_mat = scipy.io.loadmat(file_path)

            # 尝试寻找 Y (标签)
            Y = None
            for key in ['Y', 'label', 'gnd', 'labels']:
                if key in raw_mat:
                    Y = raw_mat[key].flatten()
                    break

            if Y is None:
                print(f"Skipping {dataset_name}: Label 'Y' not found.")
                continue

            # 确定 K (聚类数)
            n_cluster = len(np.unique(Y))

            # --- B. 获取 BPs (基聚类矩阵) ---
            BPs = None

            # 情况 1: 文件中已经包含 BPs
            if 'BPs' in raw_mat:
                print(f"   - Mode: Pre-computed BPs found.")
                BPs = raw_mat['BPs']
                # 处理 MATLAB 1-based 索引
                if np.min(BPs) == 1:
                    BPs = BPs - 1

            # 情况 2: 文件只有 X，需要生成
            else:
                print(f"   - Mode: Raw data found. Generating BPs using {generator_method}...")
                # 使用你的 io 模块加载 X (处理 v7.3 等复杂情况)
                X, _ = pce.io.load_mat(file_path)

                if X is None:
                    print(f"Skipping {dataset_name}: Neither 'BPs' nor 'X' found.")
                    continue

                # 调用生成器
                # 注意：这里假设你的生成器通过 kwargs 接收参数，或者固定参数
                # 如果你的 generator 接口不同，需在此调整
                BPs, _ = generator_func(X, Y, nBase=n_base, seed=seed)

            # --- C. 运行集成 (Consensus) ---
            print(f"   - Running Consensus: {consensus_method}...")
            # 调用集成算法
            labels, _ = consensus_func(BPs, Y)  # 假设 Y 只是透传，主要用到 K

            # --- D. 评估 (Evaluation) ---
            print(f"   - Evaluating...")
            res = pce.metrics.evaluation_batch(labels, Y)

            # --- E. 保存 (Saving) ---
            save_name = f"{dataset_name}_result.csv"
            save_path = output_path / save_name

            pce.io.save_csv(res, str(save_path))
            # print(f"   - Saved to: {save_name}")

        except Exception as e:
            print(f"!!! Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()  # 打印详细报错方便调试

    print("\nBatch processing completed.")
