import numpy as np
from scipy.spatial.distance import cdist
from .litekmeans_core import litekmeans_core


def bagkmeans_core(X, n_clusters, subsample_ratio=0.8, maxiter=100, replicates=1, seed=2026):
    """
    Bagging (Subsampling) K-Means Core (Single Run).

    Principle:
    1. Randomly sample a subset of samples (Subsampling).
    2. Run K-Means on the subsample to get cluster centers.
    3. Assign full data to the nearest cluster centers (Nearest Centroid Assignment).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The full data matrix.
    n_clusters : int
        Number of clusters.
    subsample_ratio : float
        Ratio of samples to select (0 < ratio <= 1). Default 0.8.
    seed : int, optional
        Random seed for subsampling and kmeans initialization.

    Returns
    -------
    full_labels : ndarray
        Cluster labels for the FULL dataset (0-based).
    sampled_indices : ndarray
        Indices of the samples used for training (the bag).
    """
    n_samples = X.shape[0]

    # Set random seed
    rng = np.random.RandomState(seed)

    # --- 1. Sample Subsampling ---
    # Determine sample size
    n_sub = int(n_samples * subsample_ratio)

    # Ensure sample size is at least equal to number of clusters, otherwise clustering is impossible
    if n_sub < n_clusters:
        n_sub = n_clusters

    # Sampling Without Replacement - more common in clustering than with replacement, avoids overlapping points affecting centroid calculation
    sampled_indices = rng.choice(n_samples, n_sub, replace=False)

    # Build subsample data
    X_sub = X[sampled_indices]

    # --- 2. Run K-Means on subsample ---
    # Set numpy seed to control initialization inside litekmeans
    if seed is not None:
        np.random.seed(seed)

    # Call litekmeans_core
    # We need 'center' (second return value) to classify full data
    _, centers, _, _, _ = litekmeans_core(X_sub, n_clusters, maxiter=maxiter, replicates=replicates)

    # --- 3. Full Sample Assignment (Assignment Step) ---
    # Calculate distance matrix from full X to centers
    # X: (N, D), centers: (K, D) -> dists: (N, K)
    dists = cdist(X, centers, metric='euclidean')

    # Get index of nearest center for each sample as label
    full_labels = np.argmin(dists, axis=1)

    return full_labels, sampled_indices
