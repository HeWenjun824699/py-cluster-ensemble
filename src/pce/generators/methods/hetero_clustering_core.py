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
    # 确保 n_clusters 是整数
    n_clusters = int(n_clusters)

    # 算法调度
    if algorithm == 'spectral':
        # Spectral Clustering: 适合非凸形状
        # eigen_solver='arpack' 通常比较稳定
        # affinity='nearest_neighbors' 在集成中通常比 rbf 表现好
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            eigen_solver='arpack',
            random_state=seed,
            n_jobs=-1  # 并行加速
        )
        label = model.fit_predict(X)

    elif algorithm in ['ward', 'average', 'complete']:
        # Agglomerative Clustering: 层次聚类
        # 注意: 这是一个确定性算法(Deterministic)，没有 random_state。
        # 集成的多样性完全来自于 n_clusters 的随机变化 (Random-k)。
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=algorithm
        )
        label = model.fit_predict(X)

    elif algorithm == 'gmm':
        # Gaussian Mixture Model: 概率模型
        model = GaussianMixture(
            n_components=n_clusters,
            random_state=seed,
            reg_covar=1e-6  # 防止协方差矩阵奇异
        )
        model.fit(X)
        label = model.predict(X)

    elif algorithm == 'kmeans':
        # 回退到 LiteKMeans
        if seed is not None:
            np.random.seed(seed)
        label = litekmeans_core(X, n_clusters)[0]

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return label
