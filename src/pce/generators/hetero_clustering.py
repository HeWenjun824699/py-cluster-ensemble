from typing import Optional, List, Union
import numpy as np

from .methods.hetero_clustering_core import hetero_clustering_core
from .utils.get_k_range import get_k_range


def hetero_clustering(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        algorithms: Union[str, List[str]] = 'auto',
        seed: int = 2026
):
    """
    Heterogeneous Ensemble Generator (异构集成生成器)

    原理: 混合使用具有不同归纳偏置(Inductive Bias)的聚类算法。
    结合了:
    1. Model Perturbation: 随机选择算法 (Spectral, Hierarchical, GMM)
    2. Parameter Perturbation: 随机 K 值 (Random-k)

    Parameters
    ----------
    algorithms : str or List[str]
        - 'auto': Randomly mix ['spectral', 'ward', 'average', 'complete', 'gmm', 'kmeans']
        - List[str]: Specify a subset, e.g. ['spectral', 'ward']
        - str: Fix one algorithm, e.g. 'spectral'
    """
    nSmp = X.shape[0]

    # --- 1. 配置算法池 ---
    if algorithms == 'auto':
        # 默认混合策略 (已补全 'complete')
        algo_pool = ['spectral', 'ward', 'average', 'complete', 'gmm', 'kmeans']
    elif isinstance(algorithms, str):
        algo_pool = [algorithms]
    else:
        algo_pool = algorithms

    # --- 2. 调用辅助函数获取 K 值范围 ---
    minCluster, maxCluster = get_k_range(n_smp=nSmp, n_clusters=nClusters, y=Y)

    # --- 3. 生成基聚类 ---
    BPs = np.zeros((nSmp, nPartitions), dtype=np.float64)

    nRepeat = nPartitions

    # 初始化随机数生成器
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    # 预先生成算法选择序列 (均匀分布)
    selected_algos = rs.choice(algo_pool, size=nRepeat)

    for iRepeat in range(nRepeat):
        current_seed = random_seeds[iRepeat]
        current_algo = selected_algos[iRepeat]

        # -------------------------------------------------
        # 步骤 A: 随机选择 K 值 (Random-k)
        # -------------------------------------------------
        np.random.seed(current_seed)

        if minCluster == maxCluster:
            iCluster = minCluster
        else:
            iCluster = np.random.randint(minCluster, maxCluster + 1)

        # -------------------------------------------------
        # 步骤 B: 运行选定的异构算法
        # -------------------------------------------------
        try:
            label = hetero_clustering_core(
                X=X,
                n_clusters=iCluster,
                algorithm=current_algo,
                seed=current_seed
            )

            # 存储结果 (转为 1-based)
            BPs[:, iRepeat] = label + 1

        except Exception as e:
            # 容错处理：某些算法(如Spectral)在特定数据下可能失败
            # 如果失败，回退到 K-Means 保证流程不中断
            # print(f"Warning: {current_algo} failed at iter {iRepeat}. Fallback to kmeans. Error: {e}")
            fallback_label = hetero_clustering_core(X, iCluster, 'kmeans', seed=current_seed)
            BPs[:, iRepeat] = fallback_label + 1

    return BPs
