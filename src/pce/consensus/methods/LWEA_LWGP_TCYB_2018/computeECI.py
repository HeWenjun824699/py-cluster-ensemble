import numpy as np

def computeECI(bcs, baseClsSegs, para_theta):
    """
    Compute Entropy of Cluster Intersection (ECI).
    
    Args:
        bcs: Globally unique cluster labels matrix.
        baseClsSegs: Sparse matrix indicating cluster membership.
        para_theta: Parameter theta.
        
    Returns:
        ECI: Vector of ECI values for each cluster.
    """
    M = bcs.shape[1]
    ETs = getAllClsEntropy(bcs, baseClsSegs)
    
    # ECI = exp(-ETs./para_theta./M)
    ECI = np.exp(-ETs / para_theta / M)
    return ECI

def getAllClsEntropy(bcs, baseClsSegs):
    """
    Get the entropy of each cluster w.r.t. the ensemble.
    """
    # In Matlab: baseClsSegs = baseClsSegs'
    # baseClsSegs was nCls x N. Transposed: N x nCls.
    # Accessing columns of the transposed matrix corresponds to accessing rows of original.
    # In Python sparse matrices (CSC/CSR), efficient access depends on format.
    # baseClsSegs is CSC (from getAllSegs).
    
    nCls = baseClsSegs.shape[0]
    Es = np.zeros(nCls)
    
    # Iterate over each global cluster
    for i in range(nCls):
        # Find data points belonging to cluster i
        # In Matlab: partBcs = bcs(baseClsSegs(:,i)~=0, :) (after transpose)
        # Here baseClsSegs is nCls x N. Row i is the cluster.
        # Get indices of data points in cluster i
        
        # Helper to get row slice efficiently
        row_start = baseClsSegs.indptr[i]
        row_end = baseClsSegs.indptr[i+1]
        
        # If CSC, indptr is for columns. baseClsSegs was created as CSC with shape (nCls, N).
        # Wait, in getAllSegs.py: csc_matrix((data, (row, col)), shape=(nCls, N)).
        # CSC is compressed sparse COLUMN.
        # So slicing rows is slow. CSR is better for row slicing.
        # Let's convert to CSR for this operation if nCls is large.
        
        # However, to strictly follow Matlab logic structure:
        # Matlab transposes: baseClsSegs = baseClsSegs'. (N x nCls)
        # Then loops i=1:nCls, accesses column i.
        # Python: let's work with the nCls x N matrix directly (row i).
        
        # Get row i
        # For CSC, getting a row is expensive. 
        # But we can convert to CSR once.
        pass
    
    baseClsSegs_csr = baseClsSegs.tocsr()
    
    for i in range(nCls):
        # indices of non-zero elements in row i
        indices = baseClsSegs_csr.indices[baseClsSegs_csr.indptr[i]:baseClsSegs_csr.indptr[i+1]]
        
        if len(indices) == 0:
            continue
            
        partBcs = bcs[indices, :]
        Es[i] = getOneClsEntropy(partBcs)
        
    return Es

def getOneClsEntropy(partBcs):
    """
    Get the entropy of one cluster w.r.t the ensemble.
    """
    E = 0
    # Iterate over columns (base clusterings)
    for i in range(partBcs.shape[1]):
        tmp = partBcs[:, i]
        # sort and unique
        uTmp, counts = np.unique(tmp, return_counts=True)
        
        if len(uTmp) <= 1:
            continue
            
        cnts = counts / np.sum(counts)
        E = E - np.sum(cnts * np.log2(cnts))
        
    return E
