import numpy as np

def compute_mca(base_cls):
    """
    Compute the microcluster based co-association matrix.
    
    Args:
        base_cls: N x M matrix of cluster labels (representing microclusters).
        
    Returns:
        S: N x N co-association matrix.
    """
    n, n_bc = base_cls.shape
    
    # Initialize S
    s = np.zeros((n, n))
    
    for k in range(n_bc):
        # Get the k-th clustering (column)
        col = base_cls[:, k]
        
        # Find unique labels in this clustering to iterate over
        unique_labels = np.unique(col)
        
        for label in unique_labels:
            if label < 0: continue # Only ignore negative labels (noise), allow 0 (valid in Python)
            
            # Find indices of objects having this label
            indices = np.where(col == label)[0]
            
            # Update co-association matrix
            # Add 1 to all pairs (i, j) where both belong to the same cluster
            # This can be vectorized with broadcasting or meshgrid
            # S[indices[:, None], indices] += 1
            
            # To strictly follow Matlab loop logic (which is O(N_subset^2)):
            for idx_tmp in indices:
                 s[idx_tmp, indices] += 1
                 
    s = s * 1.0 / n_bc
    return s
