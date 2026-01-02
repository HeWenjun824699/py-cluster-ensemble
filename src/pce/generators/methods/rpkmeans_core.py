import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from .litekmeans_core import litekmeans_core


def rpkmeans_core(X, n_clusters, projection_ratio=0.5, maxiter=100, replicates=1, seed=2026):
    """
    Random Projection K-Means Core (Single Run).

    Principle: Use Gaussian random matrix to project data into a lower-dimensional space, then execute K-Means.
    Basis: Johnson-Lindenstrauss Lemma guarantees that Euclidean distances after projection remain approximately invariant.

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

    # 1. Determine Target Dimension
    # Keep at least 1 dimension
    n_components = max(1, int(n_features * projection_ratio))

    # 2. Execute Random Projection
    # Use GaussianRandomProjection to generate projection matrix satisfying J-L Lemma
    # Its random_state controls both matrix generation and ensures reproducibility
    transformer = GaussianRandomProjection(n_components=n_components, random_state=seed)
    X_proj = transformer.fit_transform(X)

    # [Pure NumPy Alternative] If you don't want to depend on sklearn, use the following code:
    # rng = np.random.RandomState(seed)
    # R = rng.randn(n_features, n_components)
    # X_proj = X @ R

    # 3. Run K-Means in the projected space
    # Set numpy random seed to control initialization inside litekmeans
    if seed is not None:
        np.random.seed(seed)

    # Call litekmeans_core
    label = litekmeans_core(X_proj, n_clusters, maxiter=maxiter, replicates=replicates)[0]

    # Return label and projection transformer (or matrix)
    return label, transformer.components_
