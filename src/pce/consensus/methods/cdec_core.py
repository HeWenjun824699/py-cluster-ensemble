import numpy as np
from sklearn.cluster import KMeans

from .CDEC_TCSVT_2025.CDEC_lyh import CDEC_lyh
from .CDEC_TCSVT_2025.compute_Hc import compute_Hc


def cdec_core(BPi: np.ndarray, nCluster: int, lamb: float = 1e-3, gamma: float = 1e-1) -> np.ndarray:
    """
    Core implementation of CDEC (Consensus Clustering logic).

    Logic mirrors the inner loop of run_CDEC_TCSVT_2025.m:
        Hc = compute_Hc(BPi);
        now = kmeans(Hc, nCluster);
        label = CDEC_lyh(nBase, nCluster, BPi, lambda, gamma, now);

    Parameters
    ----------
    BPi : np.ndarray
        Base Partitions subset (n_samples, n_base).
    nCluster : int
        Target number of clusters.
    lamb : float
        Hyperparameter lambda (default 1e-3).
    gamma : float
        Hyperparameter gamma (default 1e-1).

    Returns
    -------
    labels : np.ndarray
        Consensus cluster labels (1D array).
    """
    nSmp = BPi.shape[0]
    nBase = BPi.shape[1]

    # 1. Compute Hc (Hypergraph/matrix representation)
    # MATLAB: Hc = compute_Hc(BPi);
    Hc, _ = compute_Hc(BPi)

    # 2. KMeans Initialization
    # MATLAB: now = kmeans(Hc, nCluster);
    # Note: MATLAB's kmeans uses random initialization.
    # The random state is controlled by the seed set in the wrapper (cdec.py)
    kmeans = KMeans(n_clusters=nCluster, n_init=10)
    initial_labels = kmeans.fit_predict(Hc)

    # 3. Run CDEC optimization
    # MATLAB: label = CDEC_lyh(nBase, nCluster, BPi, lambda, gamma, now);
    # Passing 'initial_labels' as the 'now' parameter
    labels = CDEC_lyh(nBase, nCluster, BPi, lamb, gamma, initial_labels)

    return np.array(labels).flatten()
