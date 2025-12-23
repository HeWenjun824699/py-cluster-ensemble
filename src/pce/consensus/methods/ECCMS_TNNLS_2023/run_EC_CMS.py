import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from .solver import solver

def run_EC_CMS(A, B, k, lam):
    """
    Run EC-CMS algorithm.
    
    Args:
        A: High Confidence Matrix (passed as H to solver).
        B: LWCA Matrix (passed as A to solver).
        k: List of cluster numbers.
        lam: Lambda parameter.
        
    Returns:
        results: N x len(k) matrix of cluster labels.
    """
    N = B.shape[0]
    
    if np.isscalar(k):
        k = [k]
    n_k = len(k)
    
    results = np.zeros((N, n_k))
    
    # Remove diagonal of A
    np.fill_diagonal(A, 0)
    
    C, _, _ = solver(A, B, lam)
    
    for i in range(n_k):
        K = k[i]
        
        C_no_diag = C.copy()
        np.fill_diagonal(C_no_diag, 0)
        
        # Symmetrize
        C_sym = (C_no_diag + C_no_diag.T) / 2
        
        # Distance
        # In Matlab: d = 1 - s; where s is vector of lower triangle.
        s = squareform(C_sym, checks=False)
        d = 1 - s
        d = np.clip(d, 0, None)
        
        # Hierarchical clustering
        Z = linkage(d, method='average')
        labels = fcluster(Z, t=K, criterion='maxclust')
        
        results[:, i] = labels
        
        print(f'Obtain {K} clusters.')
        
    return results
