from typing import Optional
import numpy as np

from .methods.rskmeans_core import rskmeans_core
from .utils.check_array import check_array
from .utils.get_k_range import get_k_range


def rskmeans(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        subspace_ratio: float = 0.5,
        seed: int = 2026,
        maxiter: int = 100,
        replicates: int = 1
):
    """
    Random Subspace K-Means 集成生成器

    结合了两种多样性策略：
    1. Parameter Perturbation: 随机 K 值 (Random-k)
    2. Feature Perturbation: 随机特征子空间 (Random Subspace)
    """
    nSmp = X.shape[0]

    # 【核心修改】自动处理所有格式问题
    X = check_array(X, accept_sparse=False)

    # --- 1. 调用辅助函数获取 K 值范围 ---
    # 保持与 litekmeans/cdkmeans 一致的 Random-k 逻辑
    minCluster, maxCluster = get_k_range(n_smp=nSmp, n_clusters=nClusters, y=Y)

    # --- 2. 生成基聚类 ---
    BPs = np.zeros((nSmp, nPartitions), dtype=np.float64)

    nRepeat = nPartitions

    # 初始化随机数生成器
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        current_seed = random_seeds[iRepeat]

        # -------------------------------------------------
        # 步骤 A: 随机选择 K 值 (Random-k)
        # -------------------------------------------------
        np.random.seed(current_seed)

        if minCluster == maxCluster:
            iCluster = minCluster
        else:
            iCluster = np.random.randint(minCluster, maxCluster + 1)

        # -------------------------------------------------
        # 步骤 B: 运行 Random Subspace K-Means
        # -------------------------------------------------
        # 传入 current_seed 确保特征选择和 K-Means 初始化是可复现的
        label, _ = rskmeans_core(
            X=X,
            n_clusters=iCluster,
            subspace_ratio=subspace_ratio,
            maxiter=maxiter,
            replicates=replicates,
            seed=current_seed
        )

        # 存储结果 (转为 1-based，保持与 MATLAB 习惯一致)
        BPs[:, iRepeat] = label + 1

    return BPs
