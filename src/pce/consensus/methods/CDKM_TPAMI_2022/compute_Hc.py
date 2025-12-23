import numpy as np
from scipy.sparse import csc_matrix, hstack, diags

def compute_Hc(BPs):
    """
    Compute Hc and Hc_normalized from Base Partitions (BPs).
    
    Parameters:
    BPs (numpy.ndarray): n_samples x n_base matrix of cluster labels.
    
    Returns:
    tuple: (Hc, Hc_normalized)
        Hc: Sparse matrix of concatenated one-hot encodings.
        Hc_normalized: Hc with columns normalized by the square root of their sum.
    """
    n_samples, n_base = BPs.shape
    H_list = []
    
    for i in range(n_base):
        # Retrieve the labels for the current base partition
        labels = BPs[:, i]
        
        # Handle 1-based indexing (common in MATLAB) or 0-based
        # We assume if min is 1, it's 1-based.
        if labels.min() == 1:
            cols = labels.astype(int) - 1
            n_cols = labels.max().astype(int)
        else:
            cols = labels.astype(int)
            n_cols = labels.max().astype(int) + 1
            
        rows = np.arange(n_samples)
        data = np.ones(n_samples)
        
        # Create sparse binary matrix (one-hot) for this base partition
        # Shape: (n_samples, n_clusters_in_base)
        # Using csc_matrix for efficient column stacking later
        H_i = csc_matrix((data, (rows, cols)), shape=(n_samples, n_cols))
        H_list.append(H_i)
    
    # Concatenate all one-hot matrices horizontally
    Hc = hstack(H_list)
    
    # Normalize columns: Hc_normalized = bsxfun(@rdivide, Hc, sqrt(sum(Hc, 1)))
    # MATLAB's sum(Hc, 1) sums the columns (resulting in a row vector)
    col_sums = np.array(Hc.sum(axis=0)).flatten()
    
    # Calculate normalization factors
    # Avoid division by zero
    norm_factors = np.sqrt(col_sums)
    with np.errstate(divide='ignore'):
        inv_norm_factors = 1.0 / norm_factors
    inv_norm_factors[~np.isfinite(inv_norm_factors)] = 0
    
    # Apply normalization using matrix multiplication with diagonal matrix
    D = diags(inv_norm_factors)
    Hc_normalized = Hc @ D
    
    return Hc, Hc_normalized
