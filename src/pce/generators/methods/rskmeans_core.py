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

    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)

    # --- 1. 特征子空间选择 (Feature Selection) ---
    # 确定要抽取的特征数量 (至少选取 1 个特征)
    n_sub = max(1, int(n_features * subspace_ratio))

    # 无放回随机抽取特征
    selected_features = np.random.choice(n_features, n_sub, replace=False)

    # 构建子空间数据
    X_sub = X[:, selected_features]

    # --- 2. 在子空间上运行 K-Means ---
    # 调用现有的 litekmeans_core
    # 注意：litekmeans_core 返回 (label, center, sumD, D)，我们只需要 label
    label = litekmeans_core(X_sub, n_clusters, maxiter=maxiter, replicates=replicates)[0]

    return label, selected_features
