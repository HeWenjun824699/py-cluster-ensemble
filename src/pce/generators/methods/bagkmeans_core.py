import numpy as np
from scipy.spatial.distance import cdist
from .litekmeans_core import litekmeans_core


def bagkmeans_core(X, n_clusters, subsample_ratio=0.8, maxiter=100, replicates=1, seed=2026):
    """
    Bagging (Subsampling) K-Means Core (Single Run).

    原理:
    1. 随机抽取部分样本 (Subsampling)。
    2. 在子样本上运行 K-Means 得到聚类中心。
    3. 将全量数据指派给最近的聚类中心 (Nearest Centroid Assignment)。

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

    # 设置随机种子
    rng = np.random.RandomState(seed)

    # --- 1. 样本重采样 (Subsampling) ---
    # 确定采样数量
    n_sub = int(n_samples * subsample_ratio)

    # 保证采样数至少大于等于簇数，否则无法聚类
    if n_sub < n_clusters:
        n_sub = n_clusters

    # 无放回抽样 (Without Replacement) - 在聚类中比有放回更常用，避免重叠点影响质心计算
    sampled_indices = rng.choice(n_samples, n_sub, replace=False)

    # 构建子样本数据
    X_sub = X[sampled_indices]

    # --- 2. 在子样本上运行 K-Means ---
    # 设置 numpy 种子以控制 litekmeans 内部初始化
    if seed is not None:
        np.random.seed(seed)

    # 调用 litekmeans_core
    # 我们需要 'center' (第二个返回值) 来对全量数据进行归类
    _, centers, _, _, _ = litekmeans_core(X_sub, n_clusters, maxiter=maxiter, replicates=replicates)

    # --- 3. 全量样本补全 (Assignment Step) ---
    # 计算全量 X 到 centers 的距离矩阵
    # X: (N, D), centers: (K, D) -> dists: (N, K)
    dists = cdist(X, centers, metric='euclidean')

    # 获取每个样本距离最近的中心索引作为标签
    full_labels = np.argmin(dists, axis=1)

    return full_labels, sampled_indices
