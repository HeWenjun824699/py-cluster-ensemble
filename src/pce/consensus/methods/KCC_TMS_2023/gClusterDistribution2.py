import numpy as np

def gClusterDistribution2(index, K, n):
    """
    Calculates cluster distribution for consensus partition.
    
    Args:
        index: (n,) or (n, 1) vector of cluster labels (0-based).
        K: number of clusters.
        n: number of data points.
        
    Returns:
        P: (K, 1) vector (or similar shape to match MATLAB P), cluster distribution.
    """
    index = index.flatten()
    # MATLAB: counts = hist(index, 0:K);
    # counts(1) is count of 0 (missing/outlier?), counts(2:K+1) are clusters 1..K.
    # P = counts(2:K+1) ./ (n - counts(1))
    
    # Python: index 0..K-1. -1 for missing.
    shifted_index = index + 1
    counts = np.bincount(shifted_index, minlength=K+1)
    
    # counts[0] is missing (-1). counts[1..K] are clusters 0..K-1.
    
    P = np.zeros((K, 1))
    if n - counts[0] > 0:
        P[:, 0] = counts[1:K+1] / (n - counts[0])
        
    return P
