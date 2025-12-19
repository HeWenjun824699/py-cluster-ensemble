import time
from typing import Optional

import numpy as np

from .methods.mcla_core import mcla_core
from .utils.get_k_target import get_k_target


def mcla(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
):
    """
    MCLA (Meta-Clustering Algorithm) Wrapper.
    对应 MATLAB 脚本的主逻辑：批量读取 BPs，切片运行 MCLA，评估并保存结果。
    """
    # 1.处理数据
    # 【关键】处理 MATLAB 的 1-based 索引
    # MCLA 核心算法通常也需要 0-based 索引来构建超图或矩阵
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # --- [修改点] 调用辅助函数获取唯一的 K 值 ---
    # 一行代码解决，逻辑复用
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. 实验循环
    # 准备结果容器
    labels_list = []

    # 初始化随机数生成器
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # 步骤 A: 切片 BPs
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
        # 步骤 B: 运行 MCLA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        t_start = time.time()

        try:
            # 调用核心算法
            # 注意：Python 实现通常直接接收 (n_samples, n_estimators)
            # 但 mcla_core (移植自 MATLAB) 期望 (n_clusterings, n_samples)
            label_pred = mcla_core(BPi.T, nCluster)
            label_pred = np.array(label_pred).flatten()
        except Exception as e:
            print(f"MCLA failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros_like(Y)

        labels_list.append(label_pred)
        t_cost = time.time() - t_start

    return labels_list
