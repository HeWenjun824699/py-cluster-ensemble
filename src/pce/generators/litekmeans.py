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
    Main function: Batch generate Base Partitions (BPs)
    Corresponds to the main logic of the MATLAB script
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

