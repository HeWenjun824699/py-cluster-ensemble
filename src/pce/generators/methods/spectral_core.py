import numpy as np
from sklearn import cluster


def spectral_core(X, n_clusters, seed=2026, n_init=100, affinity='nearest_neighbors', assign_labels='discretize'):
    """
    Spectral Clustering Core (Single Run).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_samples)
        Precomputed affinity/adjacency matrix.
    n_clusters : int
        Number of clusters.
    seed : int, optional
        Random seed for initialization. Default is 2026.
    n_init : int, optional
        Number of time the k-means algorithm will be run with different centroid seeds.
        Default is 100.
    affinity : str, optional
        How to construct the affinity matrix. Default is 'nearest_neighbors'.
    assign_labels : str, optional
        Strategy for assigning labels in the embedding space. Default is 'discretize'.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels (0-based).
    """
    if seed is not None:
        np.random.seed(seed)

    try:
        sc = cluster.SpectralClustering(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=n_init,
            affinity=affinity,
            assign_labels=assign_labels
        )
        labels = sc.fit_predict(X)
    except Exception as e:
        # Fallback mechanism if svd fails (rare but possible with unconnected graphs)
        print(f"Spectral clustering failed with k={n_clusters}, error: {e}")
        # Return zeros or handle gracefully
        labels = np.zeros(X.shape[0], dtype=int)

    return labels
