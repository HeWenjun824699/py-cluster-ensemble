import numpy as np
from scipy.sparse.csgraph import connected_components

from .SPCE_TNNLS_2021.Optimize import Optimize


def spce_core(BPi: np.ndarray, n_clusters: int, gamma: float = 0.5) -> np.ndarray:
    """
    SPCE core algorithm implementation.
    Corresponds to the core workflow in the MATLAB script:
    Construct Tensor (Ai) -> Optimize (S) -> Graph ConnComp (Label)

    Parameters
    ----------
    BPi : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    n_clusters : int
        Target number of clusters (k) - used in Optimize function.
    gamma : float
        Self-paced learning parameter.

    Returns
    -------
    labels : np.ndarray
        Clustering label results, shape (n_samples,).
    """
    n_samples, n_base = BPi.shape

    # -------------------------------------------------------
    # 1. Construct Co-association Matrix Tensor (Tensor Construction)
    # -------------------------------------------------------
    # MATLAB Logic:
    # Ai = zeros(nSmp, nSmp, nBase);
    # for iBase = 1:nBase
    #     YYi = sparse(ind2vec(BPi(:, iBase)')');
    #     Ai(:, :, iBase) = full(YYi * YYi');
    # end

    Ai = np.zeros((n_samples, n_samples, n_base))

    for i in range(n_base):
        labels = BPi[:, i]
        # Use broadcasting to construct binary adjacency matrix:
        # If samples u and v belong to the same cluster in the current base clustering, then Ai[u, v, i] = 1
        # This is equivalent to YYi * YYi' in MATLAB
        Ai[:, :, i] = (labels[:, None] == labels[None, :]).astype(float)

    # -------------------------------------------------------
    # 2. Self-paced Learning Optimization to Solve Consistency Matrix (Optimization)
    # -------------------------------------------------------
    # MATLAB: S = Optimize(Ai, nCluster, gamma);
    # S is the optimized consensus association matrix (Consensus Matrix)
    S = Optimize(Ai, n_clusters, gamma)

    # -------------------------------------------------------
    # 3. Generate Final Labels Based on Graph Connected Components (Graph Partitioning)
    # -------------------------------------------------------
    # MATLAB: 
    # G_temp = graph(S);
    # label = conncomp(G_temp);

    # Use scipy.sparse.csgraph to solve connected components
    # S is treated as an adjacency matrix, non-zero elements represent edges
    n_comps, labels = connected_components(csgraph=S, directed=False, return_labels=True)

    return labels.astype(int)
