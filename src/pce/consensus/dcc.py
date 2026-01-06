import time
from typing import Optional, Tuple, List, Union

import numpy as np
from sklearn.cluster import KMeans

from .utils.get_k_target import get_k_target


def _compute_consensus_matrix(BPs: np.ndarray) -> np.ndarray:
    """
    计算共识矩阵 (Co-association Matrix)。
    BPs: (n_samples, n_base_partitions)
    """
    n_samples, n_partitions = BPs.shape
    consensus_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    # 优化计算：逐个基聚类累加
    # 相比于原代码的 n_samples 循环，这里按 n_partitions 循环通常更快
    for i in range(n_partitions):
        # 取出第 i 个基聚类结果 (n_samples,)
        labels = BPs[:, i]

        # 构造连接矩阵：如果在同一个簇，则为 1
        # 利用广播机制: (N, 1) == (1, N) -> (N, N)
        # 注意：这在 N 很大(>10000)时会爆内存，如果是大图需要用稀疏矩阵
        mat = (labels[:, None] == labels[None, :]).astype(np.float32)
        consensus_matrix += mat

    # 归一化
    consensus_matrix /= n_partitions
    return consensus_matrix


def dcc(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026,
        return_matrix: bool = False
) -> Union[Tuple[List[np.ndarray], List[float]],
           Tuple[List[np.ndarray], List[float], np.ndarray]]:
    """
    DCC Consensus Strategy (K-Means on Co-association Matrix).

    Parameters (Same as CSPA standard)
    ----------
    BPs : np.ndarray (n_samples, n_estimators)
    ...
    """

    # 1. 索引处理 (1-based -> 0-based，虽然本算法不依赖数值本身，但保持习惯)
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # 获取目标聚类数
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    labels_list = []
    time_list = []
    M = None

    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        start_idx = iRepeat * nBase
        end_idx = (iRepeat + 1) * nBase

        if start_idx >= nTotalBase:
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        # 切片获取当前 Ensemble 的基聚类
        BPi = BPs[:, start_idx:end_idx]

        t_start = time.time()

        try:
            # --- DCC Consensus 核心逻辑 ---
            # 1. 计算共识矩阵 (N x N)
            M = _compute_consensus_matrix(BPi)

            # 2. 在共识矩阵上运行 K-Means
            kmeans = KMeans(n_clusters=nCluster, random_state=random_seeds[iRepeat], n_init=10)
            label_pred = kmeans.fit_predict(M)

        except Exception as e:
            print(f"DCC Consensus failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros(nSmp)

        t_cost = time.time() - t_start

        labels_list.append(label_pred)
        time_list.append(t_cost)

    if return_matrix:
        return labels_list, time_list, M
    else:
        return labels_list, time_list
