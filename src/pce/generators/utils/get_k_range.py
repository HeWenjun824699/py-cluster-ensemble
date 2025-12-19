import math
from typing import Optional, Union, Tuple, List
import numpy as np


def get_k_range(
        n_smp: int,
        n_clusters: Optional[int] = None,
        y: Optional[np.ndarray] = None
) -> Tuple[int, int]:
    """
    辅助函数：根据输入参数和数据规模，确定聚类数 K 的范围 [min_k, max_k]。

    逻辑：
    1. n_clusters 为 None (自动模式):
       - 若 Y 存在 (y is not None):
         利用 Y 的真实类别数 n_real_k。
         范围设为 [min(n_real_k, sqrt_n), max(n_real_k, sqrt_n)]。
         这样既利用了先验知识，又保留了一定的随机性范围。
       - 若 Y 不存在 (y is None):
         完全无监督，默认范围 [2, ceil(sqrt(N))]。

    2. n_clusters 为 int (固定模式):
       - 用户强制指定 K 值。
       - min_k = max_k = n_clusters。

    返回:
        (min_k, max_k) 且保证 min_k <= max_k 且 min_k >= 2
    """
    # 计算默认上限：样本数的平方根
    sqrt_n = math.ceil(math.sqrt(n_smp))

    # --- 1. 确定初始范围 ---
    if n_clusters is None:
        if y is None:
            # 情况 A: 无监督自动范围
            # 范围：[2, sqrt(N)]
            min_k = 2
            max_k = max(2, sqrt_n)
        else:
            # 情况 B: 利用 Y 信息的自动范围
            # 范围：[min(real_k, sqrt_n), max(real_k, sqrt_n)]
            n_real_k = len(np.unique(y))
            min_k = min(n_real_k, sqrt_n)
            max_k = max(n_real_k, sqrt_n)

    elif isinstance(n_clusters, int):
        # 情况 C: 用户指定固定 K
        min_k = n_clusters
        max_k = n_clusters

    else:
        raise TypeError(f"n_clusters must be None or an integer, got {type(n_clusters)}")

    # --- 2. 安全性边界处理 ---
    # 保证 K 至少为 2 (防止计算出 1 或 0)
    min_k = max(2, min_k)
    max_k = max(2, max_k)

    # 保证 min <= max (处理 min_k > max_k 的特殊边界情况)
    if min_k > max_k:
        min_k, max_k = max_k, min_k

    return int(min_k), int(max_k)
