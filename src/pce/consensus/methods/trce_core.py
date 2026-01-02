import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .TRCE_AAAI_2021.optimization import optimization


def trce_core(BPi, n_clusters, gamma):
    """
    TRCE core algorithm logic.
    Corresponds to MATLAB:
        1. Ai(:,:,iBase) = full(YY*YY')
        2. S = optimization(Ai, nCluster, gamma)
        3. label = conncomp(graph(S))

    Parameters
    ----------
    BPi : np.ndarray
        Selected base partition matrix slice, shape (n_samples, n_base)
    n_clusters : int
        Target number of clusters c
    gamma : float
        Hyperparameter gamma

    Returns
    -------
    labels : np.ndarray
        Predicted cluster labels, shape (n_samples,)
    """
    n_samples, n_base = BPi.shape

    # -------------------------------------------------
    # 1. Construct Co-association Matrix Tensor Ai (Tensor Construction)
    # -------------------------------------------------
    # MATLAB: Ai = zeros(nSmp, nSmp, nBase);
    Ai = np.zeros((n_samples, n_samples, n_base))

    for i in range(n_base):
        # Get label vector for the i-th base clustering
        vec = BPi[:, i]

        # Construct Co-association Matrix
        # Logic: If samples u and v are in the same cluster, matrix position (u, v) is 1
        # Corresponds to MATLAB: YY = ind2vec(BPi(:,iBase)')'; Ai(:,:,iBase)=full(YY*YY');

        # Use broadcasting to generate boolean matrix, convert to float
        # (N, 1) == (1, N) -> (N, N)
        mat = (vec[:, None] == vec[None, :]).astype(float)

        Ai[:, :, i] = mat

    # -------------------------------------------------
    # 2. Optimize to solve S (Optimization)
    # -------------------------------------------------
    # MATLAB: S = optimization(Ai, nCluster, gamma);
    S = optimization(Ai, n_clusters, gamma)

    # -------------------------------------------------
    # 3. Extract labels from connected components (Connected Components)
    # -------------------------------------------------
    # MATLAB:
    #   G_temp = graph(S);
    #   label = conncomp(G_temp);

    # Convert S to sparse matrix to utilize scipy's graph algorithms
    # Note: S returned by optimization is usually a similarity matrix, non-zero values represent edges
    graph_matrix = csr_matrix(S)

    # Compute connected components
    # directed=False: Treat S as an undirected graph (equivalent if S is symmetric)
    # return_labels=True: Return component ID for each node (i.e., cluster label)
    n_components, labels = connected_components(
        csgraph=graph_matrix,
        directed=False,
        return_labels=True
    )

    # If the number of connected components found is inconsistent with n_clusters,
    # This is a common phenomenon in spectral clustering or graph partitioning algorithms, usually returning the current components directly
    # Or perform subsequent processing as needed (original MATLAB code did not do extra processing)

    return labels
