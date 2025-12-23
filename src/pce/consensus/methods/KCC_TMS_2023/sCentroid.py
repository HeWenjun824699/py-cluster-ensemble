import numpy as np

def sCentroid(idx, K, r, sumKi):
    """
    Initialize centroid for each cluster.
    
    Args:
        idx: (K, r) matrix of cluster labels (0-based) for K sampled points.
        K: preferred number of clusters.
        r: number of basic partitions.
        sumKi: (r+1,) starting index vector.
    
    Returns:
        C: (K, total_clusters) centroid matrix.
    """
    # sumKi is 0-based cumulative sum of cluster counts.
    # idx contains 0-based labels.
    
    total_cols = int(sumKi[-1]) # sumKi(r+1) in MATLAB is last element
    C = np.zeros((K, total_cols))
    
    for l in range(K):
        # MATLAB: C(l, idx(l,:)+sumKi(1:r)') = 1
        # Python: idx[l, :] is (r,). sumKi[0:r] is (r,).
        # We want to set indices: idx[l, i] + sumKi[i] for i in 0..r-1
        
        indices = idx[l, :] + sumKi[:r]
        C[l, indices.astype(int)] = 1
        
    return C
