import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def spectral_clustering(W, k):
    """
    Spectral clustering algorithm.
    
    Parameters:
    W (numpy.ndarray): Affinity matrix
    k (int): Number of clusters
    
    Returns:
    numpy.ndarray: Cluster labels (1-based to match MATLAB logic often, but let's check usage)
    """
    # D = diag(1./sqrt(sum(W, 2)));
    # MATLAB sum(W, 2) sums along 2nd dimension (columns) -> row sums (column vector Nx1)
    
    row_sums = np.sum(W, axis=1)
    # Avoid division by zero
    with np.errstate(divide='ignore'):
        inv_sqrt_row_sums = 1.0 / np.sqrt(row_sums)
    inv_sqrt_row_sums[np.isinf(inv_sqrt_row_sums)] = 0
    
    D = np.diag(inv_sqrt_row_sums)
    
    # W = D * W * D;
    W_norm = D @ W @ D
    
    # [U, s, V] = svd(W);
    # MATLAB svd(W) returns U, S, V such that W = U*S*V'.
    # For symmetric W, U and V are essentially the same (up to sign).
    # Python numpy.linalg.svd returns u, s, vh. W = u @ diag(s) @ vh.
    # vh is V transposed.
    
    U, s, vh = np.linalg.svd(W_norm)
    
    # V = U(:, 1 : k);
    # In MATLAB code: V = U(:, 1:k)
    # This corresponds to the top k eigenvectors.
    # U is N x N. We take first k columns.
    
    V = U[:, :k]
    
    # V = normr(V); -> Normalize rows to unit length
    V = normalize(V, axis=1, norm='l2')
    
    # ids = kmeans(V, k, ...);
    # replicates 100
    kmeans = KMeans(n_clusters=k, n_init=100, init='k-means++')
    ids = kmeans.fit_predict(V)
    
    # MATLAB kmeans returns 1-based indices usually.
    # Python returns 0-based.
    # We will return 1-based to maintain strict correspondence with expected numerical flow if subsequent steps assume it.
    # Checking mapMicroclustersBackToObjects... it uses labels as indices.
    # If python is 0-based, we should check mapMicroclustersBackToObjects.
    
    return ids + 1
