import numpy as np
from .GTLEC_MM_2023.compute_Av import compute_Av
from .GTLEC_MM_2023.TensorEC import TensorEC
from .GTLEC_MM_2023.SpectralClustering import SpectralClustering


def gtlec_core(BPi: np.ndarray, nCluster: int, alpha: float, beta: float) -> np.ndarray:
    """
    Core implementation of Graph-Based Tensor Learning for Ensemble Clustering (GTLEC).

    Logic mirrors run_GTLEC_MM_2023.m:
      A = compute_Av(BPi);
      A_tensor = cat(3, A{:,:});
      [S,Z,obj] = TensorEC(A_tensor, nCluster, alpha, beta);
      label = SpectralClustering(abs(S)+abs(S'), nCluster);

    Parameters
    ----------
    BPi : np.ndarray
        Base Partitions subset (n_samples, n_base).
    nCluster : int
        Target number of clusters.
    alpha : float
        Regularization parameter.
    beta : float
        Regularization parameter.

    Returns
    -------
    labels : np.ndarray
        Consensus cluster labels (1D array).
    """

    # 1. Compute Association Matrices
    # MATLAB: A = compute_Av(BPi);
    # Assumption: Python compute_Av returns a list of matrices (simulating MATLAB cell array)
    A_list = compute_Av(BPi)

    # 2. Construct Tensor
    # MATLAB: A_tensor = cat(3, A{:,:});
    # Stack the list of 2D matrices along the 3rd dimension
    if isinstance(A_list, list):
        A_tensor = np.stack(A_list, axis=2)
    else:
        # Fallback if compute_Av already returns a numpy array
        A_tensor = A_list

    # 3. Tensor Ensemble Clustering Optimization
    # MATLAB: [S,Z,obj]=TensorEC(A_tensor, nCluster, alpha, beta);
    S, Z, obj = TensorEC(A_tensor, nCluster, alpha, beta)

    # 4. Spectral Clustering on Consensus Matrix
    # MATLAB: S=double(S); label = SpectralClustering(abs(S)+abs(S'), nCluster);
    # Ensure S is float
    S = S.astype(float)

    # Symmetrize S
    affinity_matrix = np.abs(S) + np.abs(S.T)

    # Get final labels
    labels = SpectralClustering(affinity_matrix, nCluster)

    # Ensure labels are flattened
    return np.array(labels).flatten()
