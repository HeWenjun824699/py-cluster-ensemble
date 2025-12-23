import time
from typing import Optional, List

import numpy as np

from .methods.mdecsc_core import mdecsc_core
from .utils.get_k_target import get_k_target


def mdecsc(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> List[np.ndarray]:
    """
    MDECSC (Multi-Diversity Ensemble Clustering via Spectral Clustering) Wrapper.
    对应 MATLAB 脚本 run_MDECSC_TCYB_2022.m 的主逻辑。

    该算法逻辑：
    1. 切片基聚类器 (BPs)
    2. 计算所有Segments (getAllSegs)
    3. 计算ECI (getECI)
    4. 计算共识矩阵 S (getLWCA)
    5. 执行谱聚类 (performSC)

    Note on Consistency with MATLAB:
    MATLAB script generates a fixed list of seeds derived from the master seed.
    This implementation replicates that behavior to ensure reproducibility per repetition.

    Parameters
    ----------
    BPs : np.ndarray
        基聚类结果矩阵 (Base Partitions), shape (n_samples, n_estimators)
    Y : np.ndarray, optional
        真实标签，用于推断聚类数 k
    nClusters : int, optional
        目标聚类簇数 k
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
    # 处理 MATLAB 的 1-based 索引
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # 获取目标聚类数
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. 实验循环配置
    labels_list = []

    # 初始化随机数生成器 (对应 MATLAB: rng(seed, 'twister'))
    rs = np.random.RandomState(seed)

    # 生成随机种子池
    # MATLAB: random_seeds = randi([0, 1000000], 1, nRepeat);
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
        # 步骤 B: 运行 MDECSC Core
        # -------------------------------------------------
        t_start = time.time()

        try:
            # 设置当前迭代的特定种子
            current_seed = random_seeds[iRepeat]
            np.random.seed(current_seed)

            # 调用封装好的核心逻辑
            # MDECSC 需要 nBase 来计算 LWCA
            label_pred = mdecsc_core(BPi, nCluster, nBase)

            # 确保输出是展平的 numpy array
            final_label = np.array(label_pred).flatten()

        except Exception as e:
            print(f"MDECSC failed on repeat {iRepeat}: {e}")
            # 发生错误时返回全零标签
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list
