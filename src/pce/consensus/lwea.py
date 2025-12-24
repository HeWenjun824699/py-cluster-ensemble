import time
from typing import Optional, List

import numpy as np

from .methods.lwea_core import lwea_core
from .utils.get_k_target import get_k_target


def lwea(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        theta: float = 10,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    LWEA (Locally Weighted Ensemble Clustering) Wrapper.
    对应 MATLAB 脚本 run_LWEA_TCYB_2018.m 的主逻辑。

    该算法通常包含以下步骤:
    1. 构建二部图 (Bipartite Graph)
    2. 计算局部权重 (Local Weights) 这里的 theta 即为相关参数
    3. 求解最终划分 (通常使用谱聚类或二部图划分)

    Parameters
    ----------
    BPs : np.ndarray
        基聚类结果矩阵 (Base Partitions), shape (n_samples, n_estimators)
    Y : np.ndarray, optional
        真实标签，用于推断聚类数 k
    nClusters : int, optional
        目标聚类簇数 k
    theta : float, default=10
        LWEA 算法中的阈值/参数 t (对应 MATLAB 中的变量 t)
    nBase : int, default=20
        每次重复实验使用的基聚类器数量
    nRepeat : int, default=10
        实验重复次数
    seed : int, default=2026
        随机种子

    Returns
    -------
    labels_list : List[np.ndarray]
        包含 nRepeat 次实验结果的列表
    """

    # 1. 数据预处理
    # 处理 MATLAB 的 1-based 索引 (最小值是 1 则减 1)
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # 获取目标聚类数
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. 实验循环配置
    labels_list = []
    time_list = []

    # 初始化随机数生成器 (对应 MATLAB: rng(seed, 'twister'))
    rs = np.random.RandomState(seed)
    # 生成 nRepeat 个随机种子 (对应 MATLAB: random_seeds = randi([0, 1000000], 1, nRepeat))
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # 步骤 A: 切片 BPs (获取当前轮次的基聚类器)
        # -------------------------------------------------
        start_idx = iRepeat * nBase
        end_idx = (iRepeat + 1) * nBase

        # 边界检查
        if start_idx >= nTotalBase:
            print(f"Warning: Not enough Base Partitions for repeat {iRepeat + 1}")
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        BPi = BPs[:, start_idx:end_idx]

        # -------------------------------------------------
        # 步骤 B: 运行 LWEA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # Explicitly set the global seed to match MATLAB's logic inside the loop
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # 调用核心算法
            # MATLAB: label = LWEA_v1(BPi, nCluster, t);
            label_pred = lwea_core(BPi, nCluster, theta)

            # 确保输出是展平的 numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"LWEA failed on repeat {iRepeat}: {e}")
            # 发生错误时返回全零标签
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
