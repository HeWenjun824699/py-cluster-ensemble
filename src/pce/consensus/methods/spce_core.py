import numpy as np
from scipy.sparse.csgraph import connected_components

from .SPCE_TNNLS_2021.Optimize import Optimize


def spce_core(BPi: np.ndarray, n_clusters: int, gamma: float = 0.5) -> np.ndarray:
    """
    SPCE 核心算法实现
    对应 MATLAB 脚本中的核心流程：
    Construct Tensor (Ai) -> Optimize (S) -> Graph ConnComp (Label)

    Parameters
    ----------
    BPi : np.ndarray
        基聚类矩阵 (Base Partitions), shape (n_samples, n_estimators)
    n_clusters : int
        目标聚类数 (k) - 传入 Optimize 函数使用
    gamma : float
        自步学习参数

    Returns
    -------
    labels : np.ndarray
        聚类标签结果, shape (n_samples,)
    """
    n_samples, n_base = BPi.shape

    # -------------------------------------------------------
    # 1. 构建共协矩阵张量 (Tensor Construction)
    # -------------------------------------------------------
    # MATLAB 逻辑:
    # Ai = zeros(nSmp, nSmp, nBase);
    # for iBase = 1:nBase
    #     YYi = sparse(ind2vec(BPi(:, iBase)')');
    #     Ai(:, :, iBase) = full(YYi * YYi');
    # end

    Ai = np.zeros((n_samples, n_samples, n_base))

    for i in range(n_base):
        labels = BPi[:, i]
        # 使用广播机制构建二值邻接矩阵:
        # 如果样本 u 和 v 在当前基聚类中属于同一簇，则 Ai[u, v, i] = 1
        # 这等价于 MATLAB 中的 YYi * YYi'
        Ai[:, :, i] = (labels[:, None] == labels[None, :]).astype(float)

    # -------------------------------------------------------
    # 2. 自步学习优化求解一致性矩阵 (Optimization)
    # -------------------------------------------------------
    # MATLAB: S = Optimize(Ai, nCluster, gamma);
    # S 是优化后的一致性关联矩阵 (Consensus Matrix)
    S = Optimize(Ai, n_clusters, gamma)

    # -------------------------------------------------------
    # 3. 基于图连通分量生成最终标签 (Graph Partitioning)
    # -------------------------------------------------------
    # MATLAB: 
    # G_temp = graph(S);
    # label = conncomp(G_temp);

    # 使用 scipy.sparse.csgraph 求解连通分量
    # S 被视为邻接矩阵，非零元素表示边
    n_comps, labels = connected_components(csgraph=S, directed=False, return_labels=True)

    return labels.astype(int)
