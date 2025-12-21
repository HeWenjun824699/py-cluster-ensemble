import os
import traceback
from typing import Optional

from pathlib import Path

# 引入库组件
from .. import io
from .. import generators
from .. import consensus
from .. import metrics


def consensus_batch(
        input_dir: str,
        output_dir: Optional[str] = None,
        save_format: str = "csv",
        consensus_method: str = 'cspa',
        generator_method: str = 'cdkmeans',
        nPartitions: int = 200,
        seed: int = 2026,
        maxiter: int = 100,
        replicates: int = 1,
        nBase: int = 20,
        nRepeat: int = 10,
        overwrite: bool = False
):
    """
    批量执行聚类集成流水线。

    Args:
        input_dir: 输入数据集目录 (.mat)
        output_dir: 结果输出目录
        save_format: 'csv', 'xlsx' (文件保存方式)
        consensus_method: 'cspa', 'mcla', 'hgpa' 等 (函数名字符串)
        generator_method: 如果数据是原始 X，使用该生成器 (如 'cdkmeans', 'litekmeans')
        nPartitions: 基聚类器数量 (仅当需要生成 BPs 时使用)
        seed: 随机种子
        maxiter: 基聚类生成中，算法最大迭代次数
        replicates: 基聚类生成中，重复聚类的次数
        nBase: 集成算法中每次使用的基聚类器的数量
        nRepeat: 实验重复次数，配合nBase使用(nBase * nRepeat = 基聚类数量)
        overwrite: 是否覆盖原来的输出数据
    """
    # 1. 准备目录
    input_path = Path(input_dir)
    # 判断是否传入输出路径
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)

    if not output_path.exists():
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # 2. 获取算法函数 (利用 getattr 动态获取)
    # generator_method, consensus_method 转小写
    generator_method = generator_method.lower()
    consensus_method = consensus_method.lower()

    try:
        save_func = getattr(io, "save_results_" + save_format)
    except AttributeError:
        raise ValueError(f"Save format '{save_format}' not found in pce.io")

    try:
        consensus_func = getattr(consensus, consensus_method)
    except AttributeError:
        raise ValueError(f"Consensus method '{consensus_method}' not found in pce.consensus")

    try:
        generator_func = getattr(generators, generator_method)
    except AttributeError:
        raise ValueError(f"Generator method '{generator_method}' not found in pce.generators")

    # 3. 遍历文件
    mat_files = list(input_path.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return

    print(f"\nFound {len(mat_files)} datasets. Starting batch process with [{consensus_method}]...")

    for file_path in mat_files:
        dataset_name = file_path.stem  # 获取文件名（不含后缀）

        # 检测输出文件是否存在
        save_path = output_path / f"{consensus_method}_result" / f"{dataset_name}_{consensus_method}_result.{save_format}"
        if not overwrite and save_path.exists():
            print(f"    - Skipping: {save_path} already exists.")
            continue

        # 如果不跳过，则开始处理
        print(f"\n>>> Processing: {dataset_name}")

        try:
            BPs = None
            Y = None

            # --- A & B. 数据加载与探测 (核心修改) ---
            try:
                # 方案 1: 优先尝试直接加载 BPs 和 Y
                # load_mat_BPs_Y 会自动处理 1-based 索引问题
                print(f"    - Attempting to load pre-computed BPs...")
                BPs, Y = io.load_mat_BPs_Y(file_path)
                print(f"    - Success: Pre-computed BPs found.")

            except IOError:
                # 方案 2: 如果找不到 BPs (IOError)，则回退尝试加载 X 并现场生成
                print(f"    - BPs not found. Fallback: Loading raw data (X)...")

                # 如果这里也失败 (如文件损坏或无 X 无 Y)，会抛出 IOError 被外层 catch
                X, Y = io.load_mat_X_Y(file_path)

                print(f"    - Generating BPs using {generator_method}...")

                # 运行基聚类生成器
                BPs = generator_func(X, Y, nPartitions=nPartitions, seed=seed, maxiter=maxiter, replicates=replicates)

            # --- C. 运行集成 (Consensus) ---
            print(f"    - Running Consensus: {consensus_method}...")
            labels = consensus_func(BPs, Y, nBase=nBase, nRepeat=nRepeat, seed=seed)

            # --- D. 评估 (Evaluation) ---
            print(f"    - Evaluating...")
            res = metrics.evaluation_batch(labels, Y)

            # --- E. 保存 (Saving) ---
            save_name = f"{dataset_name}_{consensus_method}_result.{save_format}"
            save_path = output_path / f"{consensus_method}_result" / save_name
            save_func(res, str(save_path))
            print(f"    - Saved to: {save_name}")

        except Exception as e:
            # 捕获所有异常（包括 load_mat_X_Y 失败的情况）
            print(f"!!! Error processing {dataset_name}: {e}")
            # 打印堆栈以便调试
            # traceback.print_exc()

    print("\nBatch processing completed.")
