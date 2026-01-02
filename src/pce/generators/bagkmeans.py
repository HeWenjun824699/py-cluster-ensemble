from typing import Optional
import numpy as np

from .methods.bagkmeans_core import bagkmeans_core
from .utils.check_array import check_array
from .utils.get_k_range import get_k_range


def bagkmeans(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        subsample_ratio: float = 0.8,
        seed: int = 2026,
        maxiter: int = 100,
        replicates: int = 1
):
    """
    Bagging K-Means Ensemble Generator (Resampling Strategy)

    Combines two diversity strategies:
    1. Parameter Perturbation: Random K value (Random-k)
    2. Data Perturbation: Sample Resampling (Bagging/Subsampling)

    This method significantly improves the noise resistance and robustness of the ensemble by training on different data subsets.
    """
    nSmp = X.shape[0]

    # [Core Modification] Automatically handle all format issues
    X = check_array(X, accept_sparse=False)

    # --- 1. Call helper function to get K value range ---
    # Keep Random-k logic consistent with litekmeans/cdkmeans/rskmeans/rpkmeans
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
        # Step B: Run Bagging K-Means
        # -------------------------------------------------
        # Pass current_seed to ensure sampling and initialization are reproducible
        label, _ = bagkmeans_core(
            X=X,
            n_clusters=iCluster,
            subsample_ratio=subsample_ratio,
            maxiter=maxiter,
            replicates=replicates,
            seed=current_seed
        )

        # Store results (convert to 1-based, consistent with MATLAB convention)
        BPs[:, iRepeat] = label + 1

    return BPs
