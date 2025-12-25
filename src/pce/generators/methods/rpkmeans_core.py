import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from .litekmeans_core import litekmeans_core


def rpkmeans_core(X, n_clusters, projection_ratio=0.5, maxiter=100, replicates=1, seed=2026):
    """
    Random Projection K-Means Core (Single Run).

    原理: 利用高斯随机矩阵将数据投影到低维空间，然后执行 K-Means。
    依据: Johnson-Lindenstrauss Lemma 保证了投影后的欧氏距离近似不变。

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix.
    n_clusters : int
        Number of clusters.
    projection_ratio : float
        Ratio of target components to original features (0 < ratio <= 1). 
        Default 0.5.
    seed : int, optional
        Random seed for projection matrix generation and kmeans initialization.

    Returns
    -------
    label : ndarray
        Cluster labels (0-based).
    projection_matrix : ndarray or object
        The components (matrix) used for projection.
    """
    n_samples, n_features = X.shape

    # 1. 确定目标维度 (Target Dimension)
    # 至少保留 1 个维度
    n_components = max(1, int(n_features * projection_ratio))

    # 2. 执行随机投影 (Random Projection)
    # 使用 GaussianRandomProjection 生成符合 J-L 引理的投影矩阵
    # 它的 random_state 既控制矩阵生成，也保证结果可复现
    transformer = GaussianRandomProjection(n_components=n_components, random_state=seed)
    X_proj = transformer.fit_transform(X)

    # [纯 NumPy 替代方案] 如果不想依赖 sklearn，可以使用以下代码：
    # rng = np.random.RandomState(seed)
    # R = rng.randn(n_features, n_components)
    # X_proj = X @ R

    # 3. 在投影后的空间上运行 K-Means
    # 设置 numpy 随机种子以控制 litekmeans 内部的初始化
    if seed is not None:
        np.random.seed(seed)

    # 调用 litekmeans_core
    label = litekmeans_core(X_proj, n_clusters, maxiter=maxiter, replicates=replicates)[0]

    # 返回 label 和 投影变换器(或矩阵)
    return label, transformer.components_
