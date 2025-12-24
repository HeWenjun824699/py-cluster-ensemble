import time
from typing import Optional, List, Tuple

import numpy as np

from .methods.cdkm_core import compute_Hc, Y_Initialize
from .methods.cdkm_core import cdkm_core
from .utils.get_k_target import get_k_target


def cdkm(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        nInnerRepeat: int = 5,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    CDKM (Consensus Clustering via Discrete Kernel K-Means) Wrapper.
    对应 MATLAB 脚本 run_CDKM_TPAMI_2022.m 的主逻辑。

    该算法包含双层循环结构：
    1. 外层循环：切片基聚类器 (BPs)
    2. 内层循环：多次初始化运行 CDKM_fast，取目标函数最大值对应的结果

    Note on Consistency with MATLAB:
    The provided MATLAB implementation of 'Y_Initialize' contains a hardcoded random seed (rng(2024)), 
    which causes the inner loop initialization to be identical across iterations. 
    This Python implementation intentionally diverges from that specific behavior by respecting 
    the varying seeds generated in the wrapper, allowing for proper exploration of initializations 
    as likely intended by the algorithm design.

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
        实验重复次数 (外层循环)
    nInnerRepeat : int, default=5
        内层循环次数，用于择优 (对应 MATLAB logic: n_inner_repeat = 5)
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
    time_list = []

    # 初始化随机数生成器 (对应 MATLAB: rng(seed, 'twister'))
    rs = np.random.RandomState(seed)

    # 生成随机种子池
    # MATLAB: random_seeds = randi([0, 1000000], 1, nRepeat * nRepeat);
    # 注意：MATLAB 代码中生成了 nRepeat*nRepeat 个种子，但在索引时逻辑是:
    # (iRepeat-1) * nRepeat + inner_repeat
    # 为了保持逻辑一致性，这里生成相同数量的种子
    total_seeds_needed = nRepeat * nRepeat
    random_seeds = rs.randint(0, 1000001, size=total_seeds_needed)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # 步骤 A: 切片 BPs (获取当前轮次的基聚类器)
        # -------------------------------------------------
        # MATLAB logic: idx = (iRepeat - 1) * nBase + 1 : iRepeat * nBase;
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
        # 步骤 B: 运行 CDKM
        # -------------------------------------------------
        t_start = time.time()

        try:
            # 1. 构建超图关联矩阵 Hc
            # MATLAB: Hc = compute_Hc(BPi);
            Hc, _ = compute_Hc(BPi)

            best_label = np.zeros(nSmp, dtype=int)
            obj_all = -float('inf')

            t_start = time.time()

            # 2. 内层循环择优
            # MATLAB: for inner_repeat = 1:n_inner_repeat
            for inner_repeat in range(nInnerRepeat):
                # 确定当前内层循环的种子
                # MATLAB logic: rng(random_seeds( (iRepeat-1) * nRepeat + inner_repeat ));
                # 注意：Python 索引从 0 开始，MATLAB 从 1 开始，需仔细对齐
                seed_idx = iRepeat * nRepeat + inner_repeat
                if seed_idx >= len(random_seeds):
                    # fallback if inner logic exceeds pre-gen seeds
                    current_seed = rs.randint(0, 1000001)
                else:
                    current_seed = random_seeds[seed_idx]

                # 设置当前迭代的全局种子
                np.random.seed(current_seed)

                # 初始化
                # MATLAB: [~, label_0] = Y_Initialize(nSmp, nCluster);
                _, label_0 = Y_Initialize(nSmp, nCluster)

                # 核心优化
                # MATLAB: [label, iter_num, obj_max] = CDKM_fast(Hc', label_0, nCluster);
                # 注意: MATLAB 传入了 Hc' (转置)。
                # 这里假设 Python 的 cdkm_fast 内部处理维度，或者我们在这里转置。
                # 通常 sklearn 风格数据是 (n_samples, n_features)。
                # 如果 Hc 是 (n_samples, n_hyperedges)，则不需要转置。
                # 如果 Hc 是 (n_hyperedges, n_samples)，则需要 Hc.T。
                # 此处保持传入 Hc，具体由 cdkm_core 决定如何处理。
                label_pred, _, obj_history = cdkm_core(Hc.T, label_0, nCluster)

                # 获取最终目标函数值
                # MATLAB: if obj_max(end) > obj_all
                current_obj = obj_history[-1] if isinstance(obj_history, (list, np.ndarray)) else obj_history

                if current_obj > obj_all:
                    obj_all = current_obj
                    best_label = label_pred.copy()

            # 确保输出是展平的 numpy array
            final_label = np.array(best_label).flatten()

        except Exception as e:
            print(f"CDKM failed on repeat {iRepeat}: {e}")
            # 发生错误时返回全零标签
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
