import numpy as np
from .litekmeans_core import litekmeans_core


def rskmeans_core(X, n_clusters, subspace_ratio=0.5, maxiter=100, replicates=1, seed=None):
    """
    Random Subspace K-Means Core (Single Run).

    Ref: Fred & Jain, "Combining Multiple Clusterings Using Evidence Accumulation", TPAMI 2005.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix.
    n_clusters : int
        Number of clusters.
    subspace_ratio : float
        Ratio of features to select (0 < ratio <= 1). Default 0.5.
    seed : int, optional
        Random seed for feature selection and kmeans initialization.

    Returns
    -------
    label : ndarray
        Cluster labels (0-based).
    selected_features : ndarray
        Indices of features used in this run.
    """
    n_samples, n_features = X.shape

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # --- 1. Feature Subspace Selection ---
    # Determine the number of features to select (at least 1 feature)
    n_sub = max(1, int(n_features * subspace_ratio))

    # Randomly select features without replacement
    selected_features = np.random.choice(n_features, n_sub, replace=False)

    # Build subspace data
    X_sub = X[:, selected_features]

    # --- 2. Run K-Means on the subspace ---
    # Call existing litekmeans_core
    # Note: litekmeans_core returns (label, center, sumD, D), we only need label
    label = litekmeans_core(X_sub, n_clusters, maxiter=maxiter, replicates=replicates)[0]

    return label, selected_features
