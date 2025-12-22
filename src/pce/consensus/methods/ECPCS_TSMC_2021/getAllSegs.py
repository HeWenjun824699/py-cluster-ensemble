import numpy as np

def getAllSegs(baseCls):
    """
    Generates the cluster indicator matrix and adjusts base cluster labels to be globally unique.
    
    Corresponds to getAllSegs.m
    
    Parameters:
    -----------
    baseCls : numpy.ndarray
        (N, nBC) matrix of base cluster labels.
        
    Returns:
    --------
    bcs : numpy.ndarray
        (N, nBC) matrix of globally unique cluster labels.
    baseClsSegs : numpy.ndarray
        (TotalClusters, N) indicator matrix.
    """
    # Ensure input is numpy array
    baseCls = np.array(baseCls, dtype=int)
    N, nBC = baseCls.shape
    
    # To ensure consistency and handle arbitrary labels, remap labels in each column to 0..k-1 range
    bcs = np.zeros_like(baseCls)
    nClsList = []
    
    for i in range(nBC):
        # np.unique with return_inverse gives us indices 0..k-1
        _, inv = np.unique(baseCls[:, i], return_inverse=True)
        bcs[:, i] = inv
        nClsList.append(np.max(inv) + 1)
        
    nClsList = np.array(nClsList)
    
    # Calculate cumulative sum for offsets
    # C = cumsum(nClsOrig) in Matlab
    C = np.cumsum(nClsList)
    
    # Apply offsets to make cluster IDs globally unique
    # bcs = bsxfun(@plus, bcs,[0 C(1:end-1)]) in Matlab
    offsets = np.zeros(nBC, dtype=int)
    offsets[1:] = C[:-1]
    
    bcs = bcs + offsets
    
    total_clusters = C[-1]
    
    # Build the segment indicator matrix (One-Hot Encoding of clusters)
    # baseClsSegs: (TotalClusters, N)
    baseClsSegs = np.zeros((total_clusters, N), dtype=float)
    
    for i in range(nBC):
        # For each clustering, set 1 at the corresponding cluster index for each object
        rows = bcs[:, i] # Cluster IDs (globally unique)
        cols = np.arange(N) # Object IDs
        baseClsSegs[rows, cols] = 1.0
        
    return bcs, baseClsSegs
