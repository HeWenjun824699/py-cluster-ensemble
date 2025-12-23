import numpy as np

def computeECI(bcs, baseClsSegs, para_theta):
    """
    Compute Entropy of Cluster Information (ECI).
    
    Args:
        bcs: Offset base clusterings (N x M).
        baseClsSegs: Sparse matrix (nCls x N).
        para_theta: Parameter theta (lambda in main script).
    
    Returns:
        ECI: Array of ECI values for each cluster segment.
    """
    M = bcs.shape[1]
    ETs = getAllClsEntropy(bcs, baseClsSegs)
    ECI = np.exp(-ETs / para_theta / M)
    return ECI

def getAllClsEntropy(bcs, baseClsSegs):
    """
    Get the entropy of each cluster w.r.t. the ensemble.
    """
    # baseClsSegs is nCls x N.
    # We iterate over rows (clusters).
    
    nCls = baseClsSegs.shape[0]
    Es = np.zeros(nCls)
    
    # Convert to CSR for efficient row slicing
    baseClsSegs_csr = baseClsSegs.tocsr()
    
    for i in range(nCls):
        # Find samples (columns) that belong to cluster i
        row = baseClsSegs_csr.getrow(i)
        sample_indices = row.indices
        
        # Get subset of bcs for these samples
        partBcs = bcs[sample_indices, :]
        Es[i] = getOneClsEntropy(partBcs)
        
    return Es

def getOneClsEntropy(partBcs):
    """
    Get the entropy of one cluster w.r.t the ensemble.
    """
    E = 0
    M = partBcs.shape[1]
    
    # partBcs is Samples x M
    
    for i in range(M):
        tmp = partBcs[:, i]
        
        # Unique values and counts
        uTmp, inverse_indices = np.unique(tmp, return_inverse=True)
        
        if len(uTmp) <= 1:
            continue
            
        cnts = np.bincount(inverse_indices)
        # Filter out zero counts (should not happen for present values)
        cnts = cnts[cnts > 0]
        
        # Normalize to frequencies
        freqs = cnts / np.sum(cnts)
        
        # Entropy
        E = E - np.sum(freqs * np.log2(freqs))
        
    return E
