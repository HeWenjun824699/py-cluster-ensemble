import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .TRCE_AAAI_2021.optimization import optimization


def trce_core(BPi, n_clusters, gamma):
    """
    TRCE 核心算法逻辑
    对应 MATLAB:
        1. Ai(:,:,iBase) = full(YY*YY')
        2. S = optimization(Ai, nCluster, gamma)
        3. label = conncomp(graph(S))

    Parameters
    ----------
    BPi : np.ndarray
        选定的基聚类矩阵片段, shape (n_samples, n_base)
    n_clusters : int
        目标聚类簇数 c
    gamma : float
        超参数 gamma

    Returns
    -------
    labels : np.ndarray
        预测的聚类标签, shape (n_samples,)
    """
    n_samples, n_base = BPi.shape

    # -------------------------------------------------
    # 1. 构建共协矩阵张量 Ai (Tensor Construction)
    # -------------------------------------------------
    # MATLAB: Ai = zeros(nSmp, nSmp, nBase);
    Ai = np.zeros((n_samples, n_samples, n_base))

    for i in range(n_base):
        # 获取第 i 个基聚类的标签向量
        vec = BPi[:, i]

        # 构建共协矩阵 (Co-association Matrix)
        # 逻辑: 如果样本 u 和 v 在同一簇，则矩阵位置 (u, v) 为 1
        # 对应 MATLAB: YY = ind2vec(BPi(:,iBase)')'; Ai(:,:,iBase)=full(YY*YY');

        # 利用广播机制生成布尔矩阵，转化为 float
        # (N, 1) == (1, N) -> (N, N)
        mat = (vec[:, None] == vec[None, :]).astype(float)

        Ai[:, :, i] = mat

    # -------------------------------------------------
    # 2. 优化求解 S (Optimization)
    # -------------------------------------------------
    # MATLAB: S = optimization(Ai, nCluster, gamma);
    S = optimization(Ai, n_clusters, gamma)

    # -------------------------------------------------
    # 3. 连通分量提取标签 (Connected Components)
    # -------------------------------------------------
    # MATLAB:
    #   G_temp = graph(S);
    #   label = conncomp(G_temp);

    # 将 S 转换为稀疏矩阵以利用 scipy 的图算法
    # 注意：optimization 返回的 S 通常是相似度矩阵，非零值代表边
    graph_matrix = csr_matrix(S)

    # 计算连通分量
    # directed=False: 视 S 为无向图 (S 对称时等价)
    # return_labels=True: 返回每个节点的组件 ID (即聚类标签)
    n_components, labels = connected_components(
        csgraph=graph_matrix,
        directed=False,
        return_labels=True
    )

    # 如果找到的连通分量数与 n_clusters 不一致，
    # 这是谱聚类或图划分算法的常见现象，通常直接返回当前分量作为结果
    # 或者根据需求进行后续处理（原 MATLAB 代码未做额外处理）

    return labels
