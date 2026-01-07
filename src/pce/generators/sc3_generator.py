from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from .methods.sc3_generator_core import calculate_distance
from .methods.sc3_generator_core import transformation
from .utils.check_array import check_array
from .utils.get_k_range import get_k_range


def sc3_generator(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        d_region_min: float = 0.04,
        d_region_max: float = 0.07,
        seed: int = 2026,
        maxiter: int = 100,
        n_init: int = 10,
        metric: Optional[str] = None,
        trans_method: Optional[str] = None,
        n_eigen: Optional[int] = None
) -> np.ndarray:
    """
    SC3-based Base Partition Generator.

    Generates partitions by running K-Means on different transformations
    (PCA/Laplacian) of different distance matrices (Euclidean/Pearson/Spearman)
    using varying subsets of eigenvectors (dimensions).

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    Y : np.ndarray, optional, default=None
        True labels (not used in generation, kept for API consistency).
    nClusters : int, optional, default=None
        Target cluster count k. If None, it is estimated or randomly selected
        within a range derived from the number of samples.
    nPartitions : int, default=200
        Number of base partitions to generate.
    d_region_min : float, default=0.04
        Minimum fraction of eigenvectors to use (SC3 parameter).
    d_region_max : float, default=0.07
        Maximum fraction of eigenvectors to use (SC3 parameter).
    seed : int, default=2026
        Random seed for reproducibility.
    maxiter : int, default=100
        Maximum iterations for K-Means.
    n_init : int, default=10
        Number of initializations for K-Means.
    metric : str, optional, default=None
        Distance metric to use ('euclidean', 'pearson', 'spearman').
        If None, selected randomly for each partition.
    trans_method : str, optional, default=None
        Transformation method to use ('pca', 'laplacian').
        If None, selected randomly for each partition.
    n_eigen : int, optional, default=None
        Number of eigenvectors (dimensions) to use.
        If None, selected randomly within the range defined by d_region_min/max.

    Returns
    -------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, nPartitions).
        Labels are 1-based.
    """

    n_samples = X.shape[0]

    # [Core Modification] Automatically handle all format issues
    X = check_array(X, accept_sparse=False)

    minCluster, maxCluster = get_k_range(n_smp=n_samples, n_clusters=nClusters, y=Y)

    # --- 1. Pre-calculate parameter space (Distances & Transformations) ---
    # SC3 computes these once and reuses them.
    metrics = ['euclidean', 'pearson', 'spearman']
    trans_methods = ['pca', 'laplacian']

    # Store computed transformations to avoid re-computation
    # Key: (metric, trans_method) -> Value: Eigenvectors matrix
    transform_cache = {}

    # --- 2. Determine Dimension Range (d) ---
    # SC3 logic: range of eigenvectors to use
    min_dim = int(np.floor(d_region_min * n_samples))
    max_dim = int(np.ceil(d_region_max * n_samples))
    min_dim = max(1, min_dim)
    max_dim = max(min_dim, max_dim)
    possible_dims = list(range(min_dim, max_dim + 1))

    # --- 3. Generate Partitions ---
    BPs = np.zeros((n_samples, nPartitions), dtype=np.float64)

    rng = np.random.RandomState(seed)
    # Seeds for each partition to ensure reproducibility
    partition_seeds = rng.randint(0, 1000001, size=nPartitions)

    for i in range(nPartitions):
        current_seed = partition_seeds[i]
        iter_rng = np.random.RandomState(current_seed)

        # Select k
        np.random.seed(current_seed)
        if minCluster == maxCluster:
            iCluster = minCluster
        else:
            iCluster = np.random.randint(minCluster, maxCluster + 1)

        # A. Select parameters (Fixed if provided, Random otherwise)
        curr_metric = metric if metric is not None else iter_rng.choice(metrics)
        curr_trans = trans_method if trans_method is not None else iter_rng.choice(trans_methods)
        curr_d = n_eigen if n_eigen is not None else iter_rng.choice(possible_dims)

        # B. Retrieve or Calculate Transformation
        cache_key = (curr_metric, curr_trans)
        if cache_key not in transform_cache:
            try:
                dist_mat = calculate_distance(X, metric=curr_metric)
                trans_mat = transformation(dist_mat, method=curr_trans)
                transform_cache[cache_key] = trans_mat
            except Exception as e:
                # Fallback: use raw data if transformation fails
                transform_cache[cache_key] = X

        trans_mat = transform_cache[cache_key]

        # C. Slice Eigenvectors (First d dimensions)
        # Ensure d doesn't exceed matrix dimensions
        d_safe = min(curr_d, trans_mat.shape[1])
        X_subset = trans_mat[:, :d_safe]

        # D. Run K-Means
        try:
            km = KMeans(n_clusters=iCluster, n_init=n_init, max_iter=maxiter,
                        random_state=iter_rng)
            labels = km.fit_predict(X_subset)
            # Store as 1-based labels
            BPs[:, i] = labels + 1
        except Exception as e:
            # Fallback in case of convergence failure
            BPs[:, i] = np.zeros(n_samples)

    return BPs
