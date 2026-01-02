from typing import Optional

import numpy as np

from .utils.check_array import check_array
from .utils.get_k_range import get_k_range
from .methods.litekmeans_core import litekmeans_core


def litekmeans(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        seed: int = 2026,
        maxiter: int = 100,
        replicates: int = 1
):
    """
    Batch generate Base Partitions (BPs) using the fast LiteKMeans algorithm.

    This generator implements the 'Random-k' strategy to ensure diversity.
    It automatically determines the cluster range based on the sample size
    and ground truth labels if provided.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : np.ndarray, optional
        True labels of shape (n_samples,). If `nClusters` is None, `Y` is
        used to infer a reasonable K-range: [min(K_real, sqrt(N)), max(K_real, sqrt(N))].
    nClusters : int, optional
        Target number of clusters. If provided, all base partitions will use
        this fixed K. If None, K is randomly chosen for each partition.
    nPartitions : int, default=200
        Number of base partitions to generate (number of columns in BPs).
    seed : int, default=2026
        Random seed for reproducibility. It controls the sub-seeds for
        each internal KMeans run and the Random-k selection.
    maxiter : int, default=100
        Maximum number of iterations for each LiteKMeans run.
    replicates : int, default=1
        Number of times LiteKMeans is run with different centroid seeds
        per partition; the best one (lowest dist) is kept.

    Returns
    -------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, nPartitions).
        Labels are 1-based (MATLAB compatible).
    """

    nSmp = X.shape[0]

    # [Core Modification] Automatically handle all format issues
    X = check_array(X, accept_sparse=False)

    # # Original nClusters logic
    # nCluster = len(np.unique(Y))
    #
    # # Calculate K value range (minCluster, maxCluster)
    # # Corresponds to MATLAB: min(nCluster, ceil(sqrt(nSmp)))
    # sqrt_n = math.ceil(math.sqrt(nSmp))
    # minCluster = min(nCluster, sqrt_n)
    # maxCluster = max(nCluster, sqrt_n)

    # --- 1. Call helper function to get K value range ---
    minCluster, maxCluster = get_k_range(n_smp=nSmp, n_clusters=nClusters, y=Y)

    # --- 2. Generate base partitions ---
    BPs = np.zeros((nSmp, nPartitions), dtype=np.float64)

    nRepeat = nPartitions

    # Initialize random number generator (Corresponds to MATLAB: seed = 2026; rng(seed))
    # We first generate 200 random seeds to control each iteration
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        current_seed = random_seeds[iRepeat]

        # -------------------------------------------------
        # Step A: Randomly select K value
        # -------------------------------------------------
        np.random.seed(current_seed)

        if minCluster == maxCluster:
            iCluster = minCluster
        else:
            iCluster = np.random.randint(minCluster, maxCluster + 1)

        # -------------------------------------------------
        # Step B: Run LiteKMeans
        # -------------------------------------------------
        np.random.seed(current_seed)

        # Call litekmeans
        label = litekmeans_core(X, iCluster, maxiter=maxiter, replicates=replicates)[0] + 1

        BPs[:, iRepeat] = label

    return BPs

