import math
import numpy as np

from .methods.litekmeans_core import litekmeans_core
from .methods.cdkm_fast_core import cdkm_fast_core


def cdkmeans(X, Y, nBase: int = 200, seed: int = 2024, maxiter: int = 100, replicates: int = 1):
    """
    主函数：批量生成基聚类 (Base Partitions)
    对应 MATLAB 脚本的主逻辑
    """
    nSmp = X.shape[0]
    nCluster = len(np.unique(Y))

    # 计算 K 值范围 (minCluster, maxCluster)
    # 对应 MATLAB: min(nCluster, ceil(sqrt(nSmp)))
    sqrt_n = math.ceil(math.sqrt(nSmp))
    minCluster = min(nCluster, sqrt_n)
    maxCluster = max(nCluster, sqrt_n)

    # --- 3. 生成基聚类 ---
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

    return BPs, Y

