import numpy as np
from scipy.sparse import hstack, csc_matrix, diags

def compute_Hc(bps):
    """
    Compute the Hc matrix and its normalized version.
    
    Args:
        bps (np.ndarray): Base partition matrix of shape (n_samples, n_base).
                          
    Returns:
        tuple: (Hc, Hc_normalized)
            Hc: Sparse matrix of shape (n_samples, n_total_clusters)
            Hc_normalized: Hc with columns normalized by sqrt(sum(col)).
    """
    n_samples, n_base = bps.shape
    h_list = []
    
    for i in range(n_base):
        labels = bps[:, i]
        
        # Handle 1-based indexing if present (common in MATLAB ports)
        # We shift to 0-based for internal sparse matrix construction
        if np.min(labels) == 1:
             col = labels.astype(int) - 1
        else:
             col = labels.astype(int)
        
        # MATLAB's ind2vec(ind) creates a matrix where rows correspond to indices in 'ind'.
        # Transposing it means columns correspond to indices (clusters).
        # We ensure we cover all clusters up to max(col).
        
        row = np.arange(n_samples)
        
        data = np.ones(n_samples)
        n_cls = np.max(col) + 1
        
        # Create sparse matrix (N, n_cls)
        h_i = csc_matrix((data, (row, col)), shape=(n_samples, n_cls))
        
        h_list.append(h_i)
        
    hc = hstack(h_list)
    
    # Normalization
    # sum(Hc, 1) in MATLAB is sum of rows? 
    # Wait, in MATLAB sum(A, 1) sums along the first dimension (rows), resulting in a row vector containing column sums.
    # Yes. sqrt(sum(Hc, 1)) -> sqrt of column sums.
    
    col_sums = np.array(hc.sum(axis=0)).flatten()
    
    # Avoid divide by zero
    col_norms = np.sqrt(col_sums)
    col_norms[col_norms == 0] = 1.0
    
    # bsxfun(@rdivide, Hc, val) -> Divide each column by its norm
    # Multiply by diagonal matrix of inverse norms
    d_mat = diags(1.0 / col_norms)
    hc_normalized = hc @ d_mat
    
    return hc, hc_normalized
