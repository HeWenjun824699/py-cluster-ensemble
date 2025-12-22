import numpy as np
from scipy.sparse import csc_matrix

def getAllSegs(baseCls):
    """
    Get all clusters in the ensemble and their segmentation.
    
    Args:
        baseCls: N x M matrix of cluster labels.
                 Each column is a base clustering.
                 
    Returns:
        bcs: Modified baseCls with globally unique cluster labels (re-indexed).
        baseClsSegs: Sparse matrix (nCls x N) indicating cluster membership.
    """
    N, M = baseCls.shape
    
    # Initialize bcs (to hold globally unique labels)
    bcs = np.zeros((N, M), dtype=int)
    
    current_label_start = 0
    
    # Iterate over each base clustering to normalize and shift labels
    for i in range(M):
        col = baseCls[:, i]
        
        # unique returns sorted unique elements
        # return_inverse returns indices to reconstruct the array from unique elements
        # These indices are 0-based and contiguous: 0, 1, 2, ..., K-1
        _, inverse_indices = np.unique(col, return_inverse=True)
        
        # Number of clusters in this base clustering
        if len(inverse_indices) > 0:
            n_clusters = np.max(inverse_indices) + 1
        else:
            n_clusters = 0
        
        # Assign to bcs with shift
        bcs[:, i] = inverse_indices + current_label_start
        
        # Update shift for next column
        current_label_start += n_clusters
        
    nCls = current_label_start
    
    # Construct sparse matrix baseClsSegs
    # rows: cluster IDs (0 to nCls-1)
    # cols: data point indices (0 to N-1)
    
    # Flatten column-major to match structure if needed, or just standard flatten
    # The order matters for how 'col_ind' is constructed.
    # If we flatten F (column-major), we iterate over columns (base clusterings).
    # col 0: points 0..N-1
    # col 1: points 0..N-1
    row_ind = bcs.flatten(order='F')
    col_ind = np.tile(np.arange(N), M)
    
    data = np.ones(len(row_ind))
    
    baseClsSegs = csc_matrix((data, (row_ind, col_ind)), shape=(nCls, N))
    
    return bcs, baseClsSegs