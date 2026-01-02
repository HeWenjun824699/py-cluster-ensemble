from typing import Optional
import numpy as np

from .methods.rskmeans_core import rskmeans_core
from .utils.check_array import check_array
from .utils.get_k_range import get_k_range


def rskmeans(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        subspace_ratio: float = 0.5,
        seed: int = 2026,
        maxiter: int = 100,
        replicates: int = 1
):
    """
    Random Subspace K-Means Ensemble Generator.

    Combines 'Parameter Perturbation' (Random-k) and 'Feature Perturbation'
    (Random Subspace). Each partition is trained on a randomly selected
    subset of features, making it suitable for high-dimensional data.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    Y : np.ndarray, optional
        True labels for K-range inference.
    nClusters : int, optional
        Target cluster count or None for Random-k.
    nPartitions : int, default=200
        Number of base partitions to generate.
    subspace_ratio : float, default=0.5
        Ratio of features to be randomly sampled (0 < ratio <= 1).
        Default is 0.5 as per the classic TPAMI-2005 implementation.
    seed : int, default=2026
        Seed for controlling feature sampling and K-Means initialization.
    maxiter : int, default=100
        Maximum iterations for internal K-Means.
    replicates : int, default=1
        Number of K-Means runs per feature subspace.

    Returns
    -------
    BPs : np.ndarray
        Base Partitions matrix. Labels are 1-based.
    """

    nSmp = X.shape[0]

    # [Core Modification] Automatically handle all format issues
    X = check_array(X, accept_sparse=False)

    # --- 1. Call helper function to get K value range ---
    # Keep Random-k logic consistent with litekmeans/cdkmeans
    minCluster, maxCluster = get_k_range(n_smp=nSmp, n_clusters=nClusters, y=Y)

    # --- 2. Generate base partitions ---
    BPs = np.zeros((nSmp, nPartitions), dtype=np.float64)

    nRepeat = nPartitions

    # Initialize random number generator
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        current_seed = random_seeds[iRepeat]

        # -------------------------------------------------
        # Step A: Randomly select K value (Random-k)
        # -------------------------------------------------
        np.random.seed(current_seed)

        if minCluster == maxCluster:
            iCluster = minCluster
        else:
            iCluster = np.random.randint(minCluster, maxCluster + 1)

        # -------------------------------------------------
        # Step B: Run Random Subspace K-Means
        # -------------------------------------------------
        # Pass current_seed to ensure feature selection and K-Means initialization are reproducible
        label, _ = rskmeans_core(
            X=X,
            n_clusters=iCluster,
            subspace_ratio=subspace_ratio,
            maxiter=maxiter,
            replicates=replicates,
            seed=current_seed
        )

        # Store results (convert to 1-based, consistent with MATLAB convention)
        BPs[:, iRepeat] = label + 1

    return BPs
