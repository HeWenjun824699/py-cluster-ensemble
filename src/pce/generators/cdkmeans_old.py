import os
import math
import numpy as np
import scipy.io

from .methods.litekmeans_core import litekmeans_core
from .methods.cdkm_fast_core import cdkm_fast_core
from .utils.extract_xy import extract_xy


def cdkmeans_old(file_path, output_path=None, nBase=200, seed=2024, maxiter=100, replicates=1):
    """
    主函数：批量生成基聚类 (Base Partitions)
    对应 MATLAB 脚本的主逻辑
    """
    file_name = os.path.basename(file_path)
    data_name = os.path.splitext(file_name)[0]
    file_extension = os.path.splitext(file_name)[1]

    # 判断是否传入输出路径
    if output_path is None:
        output_path = os.path.dirname(file_path)

    # 如果输出目录不存在，创建它
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    # 构造输出文件名
    out_file_name = os.path.splitext(data_name)[0] + '_CDKM' + str(nBase) + file_extension
    out_file_path = os.path.join(output_path, out_file_name)

    # 提取数据
    try:
        X, Y = extract_xy(file_path)
        if X is None: return
    except Exception as e:
        print(f"Skipping {data_name}: {e}")
        return

    nSmp = X.shape[0]
    nCluster = len(np.unique(Y))

    # 计算 K 值范围 (minCluster, maxCluster)
    # 对应 MATLAB: min(nCluster, ceil(sqrt(nSmp)))
    sqrt_n = math.ceil(math.sqrt(nSmp))
    minCluster = min(nCluster, sqrt_n)
    maxCluster = max(nCluster, sqrt_n)

    # --- 3. 生成基聚类 ---
    if not os.path.exists(out_file_path):
        print(f"Processing {file_name}...")

        BPs = np.zeros((nSmp, nBase), dtype=np.float64)

        nRepeat = nBase

        # 初始化随机数生成器 (对应 MATLAB: seed = 2024; rng(seed))
        # 我们先生成 200 个随机种子，用于控制每一次循环
        rs = np.random.RandomState(seed)
        random_seeds = rs.randint(0, 1000001, size=nRepeat)

        for iRepeat in range(nRepeat):
            current_seed = random_seeds[iRepeat]

            # -------------------------------------------------
            # 步骤 A: 随机选择 K 值
            # -------------------------------------------------
            np.random.seed(current_seed)

            if minCluster == maxCluster:
                iCluster = minCluster
            else:
                iCluster = np.random.randint(minCluster, maxCluster + 1)

            # -------------------------------------------------
            # 步骤 B: 运行 LiteKMeans
            # -------------------------------------------------
            np.random.seed(current_seed)

            # 调用 litekmeans
            label_init = litekmeans_core(X, iCluster, maxiter=maxiter, replicates=replicates)[0]

            # -------------------------------------------------
            # 步骤 C: 优化聚类 (CDKM)
            # -------------------------------------------------
            # 输入 0-based，输出也是 0-based
            # 注意：Python 中 X 不需要转置，core 内部已经处理 X @ X.T
            label_refined, _, _ = cdkm_fast_core(X, label_init, c=iCluster)

            BPs[:, iRepeat] = label_refined + 1

        # --- 4. 保存结果 ---
        scipy.io.savemat(out_file_path, {'BPs': BPs, 'Y': Y})
        print(f"{data_name} has been completed!")

    else:
        print(f"{data_name} already exists. Skipping.")

