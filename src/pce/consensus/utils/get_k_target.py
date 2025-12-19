from typing import Optional
import numpy as np


def get_k_target(
        n_clusters: Optional[int] = None,
        y: Optional[np.ndarray] = None
) -> int:
    """
    辅助函数：确定集成算法的目标聚类数 K (k_target)。

    用于 CSPA, MCLA, HGPA 等集成算法。

    逻辑优先级：
    1. n_clusters (int): 用户显式指定，优先级最高。
    2. Y (array): 用户未指定 K，但提供了 Y，推断 K = len(unique(Y))。
    3. 报错: 既无 n_clusters 也无 Y，抛出 ValueError。

    参数:
        n_clusters: 用户输入的聚类数 (必须是 int 或 None)。
        y: 真实标签 (用于推断 K)。

    返回:
        k_target (int)
    """

    # 优先级 1: 用户显式指定 (Fixed K)
    if n_clusters is not None:
        # 【关键保护】防止用户传入 float (3.0) 或 str ("3") 或 tuple
        if not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be an integer, got {type(n_clusters)}")
        # 【额外保护】防止用户传入负数或 0
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

        return n_clusters

    # 优先级 2: 用户未指定 nClusters，但提供了 Y (兼容测试模式)
    elif y is not None:
        return len(np.unique(y))

    # 优先级 3: 既无 K 也无 Y -> 报错
    else:
        raise ValueError("n_clusters must be provided if Y is None (Unsupervised mode).")
