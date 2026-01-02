import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from .litekmeans_core import litekmeans_core


def hetero_clustering_core(X, n_clusters, algorithm='spectral', seed=None, **kwargs):
    """
    Heterogeneous Clustering Core (Single Run).

    Wrapper for various scikit-learn clustering algorithms to unify the interface.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix.
    n_clusters : int
        Number of clusters.
    algorithm : str
        Algorithm name. Options:
        - 'spectral': Spectral Clustering (Graph-based)
        - 'ward': Agglomerative Clustering with Ward linkage
        - 'average': Agglomerative Clustering with Average linkage
        - 'complete': Agglomerative Clustering with Complete linkage
        - 'gmm': Gaussian Mixture Model
        - 'kmeans': Fallback to litekmeans
    seed : int, optional
        Random seed (only for algorithms that support it, e.g., Spectral, GMM).

    Returns
    -------
    label : ndarray
        Cluster labels (0-based).
    """
    # Ensure n_clusters is an integer
    n_clusters = int(n_clusters)

    # Algorithm dispatch
    if algorithm == 'spectral':
        # Spectral Clustering: Suitable for non-convex shapes
        # eigen_solver='arpack' is usually more stable
        # affinity='nearest_neighbors' usually performs better than rbf in ensembles
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            eigen_solver='arpack',
            random_state=seed,
            n_jobs=-1  # Parallel acceleration
        )
        label = model.fit_predict(X)

    elif algorithm in ['ward', 'average', 'complete']:
        # Agglomerative Clustering: Hierarchical clustering
        # Note: This is a deterministic algorithm, no random_state.
        # Diversity of the ensemble comes entirely from random variation of n_clusters (Random-k).
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=algorithm
        )
        label = model.fit_predict(X)

    elif algorithm == 'gmm':
        # Gaussian Mixture Model: Probabilistic model
        model = GaussianMixture(
            n_components=n_clusters,
            random_state=seed,
            reg_covar=1e-6  # Prevent covariance matrix singularity
        )
        model.fit(X)
        label = model.predict(X)

    elif algorithm == 'kmeans':
        # Fallback to LiteKMeans
        if seed is not None:
            np.random.seed(seed)
        label = litekmeans_core(X, n_clusters)[0]

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return label
