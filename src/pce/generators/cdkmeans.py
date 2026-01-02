from typing import Optional

import numpy as np

from .methods.cdkm_fast_core import cdkm_fast_core
from .methods.litekmeans_core import litekmeans_core
from .utils.check_array import check_array
from .utils.get_k_range import get_k_range


def cdkmeans(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        seed: int = 2026,
        maxiter: int = 100,
        replicates: int = 1
):
    """
    Generate Base Partitions using Coordinate Descent Method for k-means.

    This method refines an initial LiteKMeans clustering using the CDKM
    optimization to increase the precision and diversity of base partitions.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.
    Y : np.ndarray, optional
        Used for inferring K range when `nClusters` is None.
    nClusters : int, optional
        Fixed cluster number or None for Random-k.
    nPartitions : int, default=200
        Number of partitions to generate.
    seed : int, default=2026
        Seed for controlling sub-seeds and K selection.
    maxiter : int, default=100
        Maximum iterations for the initial clustering stage.
    replicates : int, default=1
        Repeat initial clustering and pick the best one before CDKM refinement.

    Returns
    -------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, nPartitions).
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
        label_init = litekmeans_core(X, iCluster, maxiter=maxiter, replicates=replicates)[0]

        # -------------------------------------------------
        # Step C: Optimize clustering (CDKM)
        # -------------------------------------------------
        # Input 0-based, output is also 0-based
        # Note: X does not need to be transposed in Python, core handles X @ X.T internally
        label_refined, _, _ = cdkm_fast_core(X, label_init, c=iCluster)

        BPs[:, iRepeat] = label_refined + 1

    return BPs

