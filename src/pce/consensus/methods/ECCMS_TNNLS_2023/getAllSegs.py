import numpy as np
from scipy.sparse import csc_matrix

def getAllSegs(baseCls):
    """
    Get all segments (clusters) from the ensemble of base clusterings.
    
    Args:
        baseCls: N x M matrix where N is number of points, M is number of base clusterings.
                 Values should be 1-based cluster labels.
    
    Returns:
        bcs: Modified baseCls with offset labels.
        baseClsSegs: Sparse matrix (nCls x N) representing cluster membership.
    """
    N, M = baseCls.shape
    
    # Copy to avoid modifying original
    bcs = baseCls.copy()
    
    # Assume 1-based indexing for cluster labels as per Matlab convention implied by max() usage
    nClsOrig = np.max(bcs, axis=0)
    C = np.cumsum(nClsOrig)
    
    # Create offsets: [0, C[0], C[1], ..., C[M-2]]
    offsets = np.concatenate(([0], C[:-1]))
    
    # Add offsets to each column
    bcs = bcs + offsets
    
    nCls = C[-1]
    
    # Construct sparse matrix
    # Matlab: baseClsSegs=sparse(bcs(:), repmat([1:N]',1,M), 1, nCls, N);
    # bcs(:) is column-major flattened
    
    row_indices = bcs.flatten(order='F') - 1 # Convert to 0-based index
    col_indices = np.tile(np.arange(N), M)
    data = np.ones(len(row_indices))
    
    baseClsSegs = csc_matrix((data, (row_indices, col_indices)), shape=(nCls, N))
    
    return bcs, baseClsSegs
