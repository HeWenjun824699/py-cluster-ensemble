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
    Random Subspace K-Means Ensemble Generator

    Combines two diversity strategies:
    1. Parameter Perturbation: Random K value (Random-k)
    2. Feature Perturbation: Random Subspace
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
