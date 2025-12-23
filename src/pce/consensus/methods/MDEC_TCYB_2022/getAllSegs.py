import numpy as np

def getAllSegs(baseCls):
    """
    Corresponds to getAllSegs.m
    
    Args:
        baseCls: (N, M) numpy array of base clusterings (1-based labels expected).
    
    Returns:
        bcs: Shifted base clusterings.
        baseClsSegs: (nCls, N) binary indicator matrix.
    """
    N, M = baseCls.shape
    bcs = baseCls.copy()
    
    # nClsOrig = max(bcs,[],1);
    nClsOrig = np.max(bcs, axis=0)
    
    # C = cumsum(nClsOrig);
    C = np.cumsum(nClsOrig)
    
    # bcs = bsxfun(@plus, bcs,[0 C(1:end-1)]);
    # Shift vector: [0, C[0], C[1], ..., C[M-2]]
    shift = np.concatenate(([0], C[:-1]))
    bcs = bcs + shift
    
    # nCls = nClsOrig(end)+C(end-1);
    # In Matlab C(end-1) is the cumulative sum up to the second to last element.
    # nClsOrig(end) is the max label of the last column.
    # Effectively this is just the total number of unique clusters across all base clusterings, which is C[-1].
    nCls = C[-1]
    
    baseClsSegs = np.zeros((int(nCls), N))
    
    for i in range(M):
        if i == 0:
            startK = 1
        else:
            startK = C[i-1] + 1
        
        endK = C[i]
        
        # searchVec = startK:endK;
        # In Python range is [start, end) so we use endK + 1
        searchVec = np.arange(startK, endK + 1)
        
        # F = bsxfun(@eq,bcs(:,i),searchVec);
        # bcs[:, i] is (N,), searchVec is (k,)
        # Broadcast comparison: (N, 1) == (1, k) -> (N, k)
        col_data = bcs[:, i].reshape(-1, 1)
        F = (col_data == searchVec)
        
        # baseClsSegs(searchVec,:) = F';
        # We need to map 1-based labels in searchVec to 0-based indices for baseClsSegs
        # searchVec indices: startK-1 to endK-1
        idx_start = int(startK) - 1
        idx_end = int(endK)
        
        baseClsSegs[idx_start:idx_end, :] = F.T.astype(float)
        
    return bcs, baseClsSegs
