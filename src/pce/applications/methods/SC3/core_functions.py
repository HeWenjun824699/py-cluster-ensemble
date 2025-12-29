import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import spearmanr

def calculate_distance(data, metric='euclidean'):
    """
    Calculate distance matrix between cells (rows of data).
    
    R equivalent: calculate_distance in CoreFunctions.R
    Input data assumed to be (n_samples/cells, n_features/genes).
    R input was (n_genes, n_cells) and calculated ED2 (cols) or cor (cols).
    
    Parameters
    ----------
    data : np.ndarray
        (n_cells, n_genes)
    metric : str
        'euclidean', 'pearson', 'spearman'
        
    Returns
    -------
    np.ndarray
        (n_cells, n_cells) distance matrix.
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

def norm_laplacian(A):
    """
    Calculate Normalized Graph Laplacian.
    Matches Rcpp implementation in methods/SC3_R/src/cppFunctions.cpp.
    
    L = I - D^-0.5 * A * D^-0.5
    where A = exp(-A / max(A))
    
    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix (Distance matrix).
        
    Returns
    -------
    np.ndarray
        Laplacian matrix.
    """
    # A = exp(-A/A.max())
    max_A = np.max(A)
    if max_A == 0:
        # Avoid division by zero if all distances are 0
        A_new = np.exp(0) * np.ones_like(A) # All 1s
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

def transformation(dists, method='pca'):
    """
    Transform distance matrix.
    
    R equivalent: transformation in CoreFunctions.R
    
    Parameters
    ----------
    dists : np.ndarray
        (n_cells, n_cells) distance matrix.
    method : str
        'pca' or 'laplacian'
        
    Returns
    -------
    np.ndarray
        Transformed matrix (columns are eigenvectors/components).
        Sorted by eigenvalues.
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

def estkTW(dataset):
    """
    Estimate optimal k using Tracy-Widom theory.
    
    R equivalent: estkTW in CoreFunctions.R
    
    Parameters
    ----------
    dataset : np.ndarray
        (n_cells, n_genes) - CAUTION: R code expects (genes, cells).
        We assume input here is (n_cells, n_genes) to match Python convention.
        
    Returns
    -------
    int
        Estimated k.
    """
    # R: p <- ncol(dataset) (cells), n <- nrow(dataset) (genes)
    # Python: dataset is (cells, genes).
    p = dataset.shape[0] # cells
    n = dataset.shape[1] # genes
    
    # R: x <- scale(dataset)
    # R scale works on columns. Dataset is (genes, cells).
    # So it scales each cell (column).
    # Python: We need to scale each row (cell).
    # StandardScaler scales columns.
    scaler = StandardScaler()
    # Fit on dataset.T (genes, cells) to scale cells? 
    # Or just use axis.
    # We want: for each cell (row), mean=0, std=1.
    x_scaled = scaler.fit_transform(dataset.T).T # (cells, genes)
    
    # muTW <- (sqrt(n - 1) + sqrt(p))^2
    muTW = (np.sqrt(n - 1) + np.sqrt(p))**2
    
    # sigmaTW <- (sqrt(n - 1) + sqrt(p)) * (1/sqrt(n - 1) + 1/sqrt(p))^(1/3)
    sigmaTW = (np.sqrt(n - 1) + np.sqrt(p)) * (1/np.sqrt(n - 1) + 1/np.sqrt(p))**(1/3)
    
    # sigmaHatNaive <- tmult(x)
    # R tmult(x) = x %*% t(x) (since x is genes x cells, result is genes x genes? Wait.)
    # Let's check cppFunctions.cpp: returns x.t() * x.
    # If x is (genes, cells), x.t() is (cells, genes). x.t() * x is (cells, cells).
    # Correct.
    # Python x_scaled is (cells, genes).
    # We want (cells, cells).
    # So x_scaled * x_scaled.T
    
    sigmaHatNaive = x_scaled @ x_scaled.T
    
    # bd <- 3.273 * sigmaTW + muTW
    bd = 3.273 * sigmaTW + muTW
    
    # evals <- eigen(sigmaHatNaive, symmetric = TRUE, only.values = TRUE)$values
    # R eigen returns sorted decreasing.
    evals = eigh(sigmaHatNaive, eigvals_only=True)
    # eigh returns ascending.
    
    # k <- 0; for(i in 1:length(evals)){ if(evals[i] > bd) k++ }
    k = np.sum(evals > bd)
    
    return k

def consensus_matrix(partitions):
    """
    Calculate consensus matrix.
    
    Parameters
    ----------
    partitions : np.ndarray
        (n_samples, n_partitions) - Cluster labels.
        
    Returns
    -------
    np.ndarray
        (n_samples, n_samples) Consensus matrix.
    """
    n_samples = partitions.shape[0]
    n_partitions = partitions.shape[1]
    
    consensus = np.zeros((n_samples, n_samples))
    
    for i in range(n_partitions):
        labels = partitions[:, i]
        # Outer equality check
        # (N, 1) == (1, N) -> (N, N) bool matrix
        mat = (labels[:, None] == labels[None, :])
        consensus += mat.astype(float)
        
    consensus /= n_partitions
    return consensus
