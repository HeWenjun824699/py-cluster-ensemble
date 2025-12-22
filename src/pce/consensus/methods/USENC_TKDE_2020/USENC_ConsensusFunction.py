import numpy as np
from scipy import sparse
from Tcut_for_bipartite_graph import Tcut_for_bipartite_graph

def USENC_ConsensusFunction(baseCls, k, maxTcutKmIters=100, cntTcutKmReps=3, random_state=None):
    """
    USENC_ConsensusFunction
    
    Combine the M base clusterings in baseCls to obtain the final clustering
    result (with k clusters).
    
    Parameters:
    -----------
    baseCls : numpy.ndarray
        N x M matrix of base clusterings.
        Rows correspond to samples, columns to ensemble members.
    k : int
        Number of target clusters.
    maxTcutKmIters : int, optional (default=100)
    cntTcutKmReps : int, optional (default=3)
    random_state : int, RandomState instance or None, optional (default=None)
        
    Returns:
    --------
    labels : numpy.ndarray
        Final cluster labels.
    """
    
    N, M = baseCls.shape
    
    # Ensure baseCls is integer type
    baseCls = baseCls.astype(int)
    
    # Check if baseCls is 0-based or 1-based. 
    # The logic below "shifts" cluster IDs to be unique across columns.
    # We will assume input is 1-based (Matlab standard) or 0-based.
    # It doesn't strictly matter as long as we treat them as distinct categories.
    # However, to match Matlab logic exactly:
    # maxCls = max(baseCls);
    # for i = 1:numel(maxCls)-1
    #     maxCls(i+1) = maxCls(i+1)+maxCls(i);
    # end
    # cntCls = maxCls(end);
    # baseCls(:,2:end) = baseCls(:,2:end) + repmat(maxCls(1:end-1),N,1);
    
    # Detect if 0 is present to handle 0-based (Python) vs 1-based (Matlab) input
    is_0_based = np.min(baseCls) == 0
    max_vals = np.max(baseCls, axis=0)
    
    # Calculate counts and offsets
    # If 0-based, range 0..max -> max+1 clusters.
    # If 1-based, range 1..max -> max clusters.
    counts = max_vals + 1 if is_0_based else max_vals
    
    cum_counts = np.cumsum(counts)
    offsets = np.zeros(M, dtype=int)
    offsets[1:] = cum_counts[:-1]
    
    cntCls = cum_counts[-1]
    
    # Apply offsets
    # baseCls_shifted = baseCls + offsets
    baseCls_shifted = baseCls + offsets[np.newaxis, :]
    
    # Build the bipartite graph.
    # Matlab: B = sparse(repmat([1:N]',1,M), baseCls(:), 1, N, cntCls)
    
    row_ind = np.tile(np.arange(N), M)
    col_ind = baseCls_shifted.flatten(order='F')
    
    # Adjust for 1-based indexing if necessary for the sparse matrix construction (0-based in Scipy)
    if not is_0_based:
        col_ind = col_ind - 1
        
    shape_cols = cntCls

    data = np.ones(len(row_ind))
    
    # Check shape
    # If 1-based, max index is cntCls-1 (after -1). Size is cntCls.
    B = sparse.coo_matrix((data, (row_ind, col_ind)), shape=(N, shape_cols)).tocsr()
    
    # colB = sum(B)
    # B(:, colB==0) = []
    # Remove empty columns
    col_sums = np.array(B.sum(axis=0)).flatten()
    non_empty_cols = np.where(col_sums > 0)[0]
    
    B = B[:, non_empty_cols]
    
    # Cut the bipartite graph.
    labels = Tcut_for_bipartite_graph(B, k, maxTcutKmIters, cntTcutKmReps, random_state=random_state)
    
    return labels
