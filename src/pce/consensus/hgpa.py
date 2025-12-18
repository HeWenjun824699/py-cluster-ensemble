import time
import numpy as np

from .methods.hgpa_core import hgpa_core


def hgpa(BPs: np.ndarray, Y: np.ndarray, nBase: int = 20, nRepeat: int = 10, seed: int = 2026):
    """
    HGPA (HyperGraph Partitioning Algorithm) Wrapper.
    对应 MATLAB 脚本的主逻辑：批量读取 BPs，切片运行 HGPA，评估并保存结果。
    """
    # 1. 提取数据 (加载 BPs 和 Y)
    # 【关键】处理 MATLAB 的 1-based 索引
    # HGPA 核心算法通常基于超图，需要 0-based 索引来构建关联矩阵
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]
    nCluster = len(np.unique(Y))

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
        # 步骤 B: 运行 HGPA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        t_start = time.time()

        try:
            # 调用核心算法
            # 注意：此处保留了您 mcla_old.py 中的 .T 风格
            # 如果您的 hgpa_core 期望 (n_samples, n_estimators)，请去掉 .T
            label_pred = hgpa_core(BPi.T, nCluster)
            label_pred = np.array(label_pred).flatten()
        except Exception as e:
            print(f"HGPA failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros_like(Y)

        labels_list.append(label_pred)
        t_cost = time.time() - t_start

    return labels_list, Y

