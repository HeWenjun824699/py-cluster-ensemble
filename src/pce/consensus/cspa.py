import time
import numpy as np

from .methods.cspa_core import cspa_core


def cspa(BPs: np.ndarray, Y: np.ndarray, nBase: int = 20, nRepeat: int = 10, seed: int = 2026):
    """
    CSPA (Cluster-based Similarity Partitioning Algorithm) Wrapper.
    对应 MATLAB 脚本的主逻辑：批量读取 BPs，切片运行 CSPA，评估并保存结果。
    """
    # 1.处理数据
    # 【关键】处理 MATLAB 的 1-based 索引
    # 如果 BPs 是从 MATLAB 生成的 (litekmeans + 1)，最小值是 1
    # Python 的 cspa_core (基于矩阵运算) 需要 0-based 索引
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]
    nCluster = len(np.unique(Y))

    # 2. 实验循环
    # 准备结果容器
    # MATLAB: CSPA_result = zeros(nRepeat, nMeasure);
    labels_list = []

    # 初始化随机数生成器
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # 步骤 A: 切片 BPs
        # -------------------------------------------------
        # MATLAB: idx = (iRepeat - 1) * nBase + 1 : iRepeat * nBase;
        # Python: [start, end)
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
        # 步骤 B: 运行 CSPA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]
        # 设置当前轮次的随机种子 (控制 SpectralClustering 的初始化)
        # 注意：这里主要影响 cspa_core 内部的 kmeans/discretization

        t_start = time.time()

        try:
            # 调用核心算法
            # 注意：cspa_core 接收 (n_estimators, n_samples)
            # BPi 是 (n_samples, n_estimators), 所以需要转置
            label_pred = cspa_core(BPi.T, nCluster)
            label_pred = np.array(label_pred).flatten()
        except Exception as e:
            print(f"CSPA failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros_like(Y)

        labels_list.append(label_pred)
        t_cost = time.time() - t_start

    return labels_list, Y

