from typing import Optional

import numpy as np

from .utils.check_array import check_array
from .utils.get_k_range import get_k_range
from .methods.litekmeans_core import litekmeans_core


def litekmeans(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        seed: int = 2026,
        maxiter: int = 100,
        replicates: int = 1
):
    """
    主函数：批量生成基聚类 (Base Partitions)
    对应 MATLAB 脚本的主逻辑
    """
    nSmp = X.shape[0]

    # 【核心修改】自动处理所有格式问题
    X = check_array(X, accept_sparse=False)

    # # 原 nClusters 逻辑
    # nCluster = len(np.unique(Y))
    #
    # # 计算 K 值范围 (minCluster, maxCluster)
    # # 对应 MATLAB: min(nCluster, ceil(sqrt(nSmp)))
    # sqrt_n = math.ceil(math.sqrt(nSmp))
    # minCluster = min(nCluster, sqrt_n)
    # maxCluster = max(nCluster, sqrt_n)

    # --- 1. 调用辅助函数获取 K 值范围 ---
    minCluster, maxCluster = get_k_range(n_smp=nSmp, n_clusters=nClusters, y=Y)

    # --- 2. 生成基聚类 ---
    BPs = np.zeros((nSmp, nPartitions), dtype=np.float64)

    nRepeat = nPartitions

    # 初始化随机数生成器 (对应 MATLAB: seed = 2026; rng(seed))
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
        label = litekmeans_core(X, iCluster, maxiter=maxiter, replicates=replicates)[0] + 1

        BPs[:, iRepeat] = label

    return BPs

