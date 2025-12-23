import numpy as np
from .gClusterDistribution import gClusterDistribution
from .distance_euc import distance_euc
from .distance_kl import distance_kl
from .distance_cos import distance_cos
from .distance_lp import distance_lp
from .distance_euc_miss import distance_euc_miss
from .distance_kl_miss import distance_kl_miss
from .distance_cos_miss import distance_cos_miss
from .distance_lp_miss import distance_lp_miss

def Preprocess(IDX, U, n, r, w, utilFlag):
    """
    Preprocess for consensus clustering.
    IDX: (n, r) matrix of labels (0-based for clusters, -1 for missing).
    """
    # Ki = max(IDX) -> max cluster label per partition
    # Assuming IDX is 0-based. max(IDX) is max label. Number of clusters is max+1.
    # Note: If -1 is present, we should ignore it for max calculation.
    
    # We need Ki to be the count of clusters. If labels are 0..K-1, max is K-1. Ki = max+1.
    Ki = np.max(IDX, axis=0) + 1
    # Check if any -1 (missing)
    # If a column is all -1, Ki would be 0.
    
    sumKi = np.zeros(r + 1)
    for i in range(r):
        sumKi[i+1] = sumKi[i] + Ki[i]
        
    # binIDX = IDX + repmat(sumKi(1:r)',n,1)
    # IDX (0-based) + Start Index
    # Handle missing values: if IDX is -1, binIDX should be invalid or handled?
    # In MATLAB, missing is 0. sumKi starts at 0.
    # binIDX = 0 + offset -> offset.
    # In distance functions, we use binIDX to index C.
    # If missing, we shouldn't use binIDX or we use M to filter.
    # We calculate binIDX ignoring missing status, it will be garbage for missing entries but unused.
    
    binIDX = IDX + sumKi[:r] 
    
    # Check missing
    # In Python, we assume -1 is missing.
    missFlag = 0
    missMatrix = None
    
    if np.any(IDX < 0):
        missFlag = 1
        missMatrix = (IDX >= 0).astype(int) # 1 if present, 0 if missing
    else:
        missMatrix = np.array([])
        
    # Select distance function
    u_name = U[0].lower()
    
    if missFlag == 1:
        if u_name == 'u_c': distance = distance_euc_miss
        elif u_name == 'u_h': distance = distance_kl_miss
        elif u_name == 'u_cos': distance = distance_cos_miss
        elif u_name == 'u_lp': distance = distance_lp_miss
        else: raise ValueError('Unknown Utility Function')
    else:
        if u_name == 'u_c': distance = distance_euc
        elif u_name == 'u_h': distance = distance_kl
        elif u_name == 'u_cos': distance = distance_cos
        elif u_name == 'u_lp': distance = distance_lp
        else: raise ValueError('Unknown Utility Function')
        
    Pvector = None
    if (len(U) > 1 and U[1].lower() == 'norm') or utilFlag == 1:
        P = gClusterDistribution(IDX, Ki, n)
        
        eps = np.finfo(float).eps
        if u_name == 'u_c':
            Pvector = np.sum(P**2, axis=0)
        elif u_name == 'u_h':
             # Pvector = -sum(P.*log2(P+eps));
             Pvector = -np.sum(P * np.log2(P + eps), axis=0)
        elif u_name == 'u_cos':
            Pvector = np.sqrt(np.sum(P**2, axis=0))
        elif u_name == 'u_lp':
            p = U[2]
            Pvector = np.sum(P**p, axis=0)**(1/p)
            
    weight = None
    if len(U) > 1 and U[1].lower() == 'norm':
         # weight = w./Pvector'
         weight = w / Pvector
    else:
        weight = w
        
    return Ki, sumKi, binIDX, missFlag, missMatrix, distance, Pvector, weight
