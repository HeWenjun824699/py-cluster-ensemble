import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize

def baseline_SC(W, nCluster):
    """
    Baseline Spectral Clustering.
    
    Args:
        W: Affinity matrix (numpy array or sparse matrix)
        nCluster: Number of clusters
        
    Returns:
        H_normalized: Normalized eigenvector matrix
    """
    # D = sum(W,2);
    D = np.sum(W, axis=1)
    
    # DD = D.^(-0.5);
    # Handle division by zero or negative values if any
    with np.errstate(divide='ignore'):
        DD = np.power(D, -0.5)
    DD[np.isinf(DD)] = 0
    DD[np.isnan(DD)] = 0
    
    # L = diag(DD)*W*diag(DD);
    # Using element-wise broadcasting for diagonal multiplication if W is dense
    # If W is sparse, we might want to use sparse diag. Assuming dense for now as per simple Matlab code.
    if hasattr(W, 'toarray'): # Is sparse
        # This logic works for both dense and sparse if we construct diagonal matrices
        # But broadcasting is faster for dense: DD[:, None] * W * DD[None, :]
        # Let's stick to matrix multiplication for safety with sparse logic if needed,
        # but the Matlab code implies simple matrices.
        DD_mat = np.diag(DD)
        L = DD_mat @ W @ DD_mat
    else:
        # Dense optimization
        L = W * DD[:, np.newaxis] # Multiply rows
        L = L * DD[np.newaxis, :] # Multiply cols
        
    # L = eye(size(W,2)) - L;
    L = np.eye(W.shape[1]) - L
    
    # L = (L + L')/2;
    L = (L + L.T) / 2
    
    # [V, ~] = eigs(L, nCluster, 'SA');
    # 'SA' means Smallest Algebraic. eigsh finds largest by default.
    # 'SA' in Matlab eigs corresponds to 'SA' in scipy eigsh (if sigma is None).
    # Note: eigs in Matlab for L=I-D^-0.5WD^-0.5 (Normalized Laplacian)
    # Smallest eigenvalues of L are close to 0.
    vals, V = eigsh(L, k=nCluster, which='SA')
    
    # H = V(:, 1:nCluster);
    # scipy eigsh returns eigenvalues in ascending order for 'SA', so V is already sorted.
    H = V[:, :nCluster]
    
    # H_normalized = normr(H);
    # normr normalizes rows to have length 1.
    H_normalized = normalize(H, axis=1, norm='l2')
    
    return H_normalized
