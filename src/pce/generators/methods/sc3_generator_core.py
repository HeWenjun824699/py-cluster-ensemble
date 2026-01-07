import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def norm_laplacian(A):
    """
    Calculate Normalized Graph Laplacian.

    Matches Rcpp implementation in methods/SC3_R/src/cppFunctions.cpp.

    The formula used is:
    L = I - D^-0.5 * A * D^-0.5
    where A = exp(-A / max(A))

    Parameters
    ----------
    A : np.ndarray
        A symmetric distance matrix.

    Returns
    -------
    L : np.ndarray
        The calculated normalized Laplacian matrix.
    """
    # A = exp(-A/A.max())
    max_A = np.max(A)
    if max_A == 0:
        # Avoid division by zero if all distances are 0
        A_new = np.exp(0) * np.ones_like(A)  # All 1s
    else:
        A_new = np.exp(-A / max_A)

    # D_row = pow(sum(A), -0.5)
    D = np.sum(A_new, axis=1)

    # Handle potential zeros in D (though exp(-x) > 0 usually)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D_inv_sqrt = np.power(D, -0.5)

    # Replace infs with 0
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0

    # A.each_row() %= D_row; -> Multiply each row i by D_row[i]
    # A.each_col() %= D_col; -> Multiply each col j by D_col[j]
    # This is equivalent to D^-0.5 * A * D^-0.5
    # Broadcasting: (N, 1) * (N, N) * (1, N)

    D_diag = np.diag(D_inv_sqrt)
    # L = I - D^-0.5 * A * D^-0.5
    res = np.eye(A.shape[0]) - D_diag @ A_new @ D_diag

    return res


def calculate_distance(data, metric='euclidean'):
    """
    Calculate distance matrix between cells (rows of data).

    R equivalent: calculate_distance in CoreFunctions.R
    Input data assumed to be (n_samples/cells, n_features/genes).
    R input was (n_genes, n_cells) and calculated ED2 (cols) or cor (cols).

    Parameters
    ----------
    data : np.ndarray
        Input data matrix with shape (n_cells, n_features).
    metric : str, optional
        The distance metric to use. Options are 'euclidean', 'pearson', 'spearman'.
        Default is 'euclidean'.

    Returns
    -------
    dists : np.ndarray
        The calculated distance matrix with shape (n_cells, n_cells).
    """
    if metric == 'euclidean':
        # R uses ED2 (Euclidean distance between columns/cells).
        # We calculate Euclidean distance between rows/cells.
        return squareform(pdist(data, metric='euclidean'))

    elif metric == 'pearson':
        # R: 1 - cor(data, method="pearson")
        # R cor calculates correlation between columns (cells).
        # np.corrcoef calculates correlation between rows (if rowvar=True, default).
        # We want correlation between cells (rows).
        corr = np.corrcoef(data)
        return 1 - corr

    elif metric == 'spearman':
        # R: 1 - cor(data, method="spearman")
        # scipy.stats.spearmanr with axis=1 calculates corr between rows.
        # spearmanr returns (corr, pvalue).
        corr, _ = spearmanr(data, axis=1)
        return 1 - corr

    else:
        raise ValueError(f"Unknown metric: {metric}")


def transformation(dists, method='pca'):
    """
    Transform distance matrix.

    R equivalent: transformation in CoreFunctions.R

    Parameters
    ----------
    dists : np.ndarray
        Input distance matrix with shape (n_cells, n_cells).
    method : str, optional
        The transformation method to use. Options are 'pca' or 'laplacian'.
        Default is 'pca'.

    Returns
    -------
    transformed : np.ndarray
        The transformed matrix where columns are eigenvectors or principal components,
        sorted by eigenvalues.
    """
    if method == 'pca':
        # R: prcomp(dists, center=TRUE, scale.=TRUE)
        # R prcomp returns rotation (eigenvectors of covariance of scaled data).
        # But wait! R code: `return(t$rotation)`?
        # Let's check CoreFunctions.R:
        # t <- prcomp(dists, center = TRUE, scale. = TRUE)
        # return(t$rotation) -> Returns the Loading Matrix (Variables x PCs).
        # Input to prcomp is `dists` (N x N).
        # So it treats the N columns as variables?
        # Yes.
        # Python PCA:
        # fit(X) -> components_ (PCs x Features) -> Transpose to get (Features x PCs) equivalent to Rotation?
        # `pca.fit_transform(X)` returns `X @ V`.
        # `prcomp$x` is `X @ V`.
        # `prcomp$rotation` is `V` (Eigenvectors of Cov(X)).

        # SC3_R R code returns `t$rotation`.
        # The input is `dists` (symmetric).
        # We need the eigenvectors of the covariance of the scaled distance matrix.

        scaler = StandardScaler()
        dists_scaled = scaler.fit_transform(dists)

        # PCA in sklearn
        # We want V (eigenvectors of Cov(dists_scaled)).
        # PCA computes SVD of X. X = U S V^T.
        # Cov = V S^2 V^T / (n-1).
        # components_ attribute is V^T (n_components, n_features).
        # We want V (n_features, n_components).

        pca = PCA(n_components=None)
        pca.fit(dists_scaled)

        # pca.components_ is V^T. Shape (n_samples, n_features) because n_samples=n_features here.
        # We want V.
        return pca.components_.T

    elif method == 'laplacian':
        # R: L <- norm_laplacian(dists); l <- eigen(L); return(l$vectors[, order(l$values)])
        L = norm_laplacian(dists)

        # eigen(L) in R returns eigenvalues/vectors.
        # Default in R eigen is NOT sorted by value? No, "values are sorted in decreasing order".
        # But code says `order(l$values)`. `order` returns permutation to sort ascending?
        # default `decreasing = FALSE` in `order`.
        # So R code sorts eigenvectors by eigenvalues from Smallest to Largest.
        # This makes sense for Laplacian (Fiedler vector etc. are small eigenvalues).

        evals, evecs = eigh(L)
        # eigh returns sorted eigenvalues (ascending).
        # So we just return evecs.
        return evecs

    else:
        raise ValueError(f"Unknown transformation: {method}")
