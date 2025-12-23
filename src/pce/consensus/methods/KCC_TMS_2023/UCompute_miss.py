import numpy as np
from .gClusterDistribution2 import gClusterDistribution2

def UCompute_miss(index, U, w, C, n, r, K, sumKi, Pvector, M):
    """
    M: (n, r) boolean/binary matrix, 1 if present, 0 if missing.
    """
    Pc = gClusterDistribution2(index, K, n)
    Pci = np.zeros((K, r))
    
    # numi = sum(M) -> count of non-missing per partition
    numi = np.sum(M, axis=0) 
    
    for i in range(r):
        # Filter index by M[:, i]
        valid_indices = index[M[:, i].astype(bool)]
        if numi[i] > 0:
            Pci[:, i] = gClusterDistribution2(valid_indices, K, numi[i]).flatten()
            
    ratioi = numi / n
    wei = w * ratioi # Element-wise
    
    Cmatrix = np.zeros((K, r))
    util = np.zeros(2)
    
    u_type = U[0].lower()
    u_norm = U[1].lower() if len(U) > 1 else 'std'
    eps = np.finfo(float).eps
    
    if u_type == 'u_c':
        for i in range(r):
            start = int(sumKi[i])
            end = int(sumKi[i+1])
            tmp = C[:, start:end]
            Cmatrix[:, i] = np.sum(tmp**2, axis=1)
            
        term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
        
        if u_norm == 'std':
            util[0] = np.dot(term1, wei)
            util[1] = util[0] / np.sum(Pc**2)
        else:
            util[0] = np.dot(term1, wei / Pvector)
            util[1] = np.dot(term1, wei / np.sqrt(Pvector)) / np.sqrt(np.sum(Pc**2))
            
    elif u_type == 'u_h':
        for i in range(r):
            start = int(sumKi[i])
            end = int(sumKi[i+1])
            tmp = C[:, start:end]
            Cmatrix[:, i] = np.sum(tmp * np.log2(tmp + eps), axis=1)
            
        # For U_H, MATLAB: (sum(...) + Pvector) * wei.
        # Wait, UCompute.m had (sum - Pvector).
        # UCompute_miss.m has (sum + Pvector).
        # Let's verify UCompute_miss.m content from memory.
        # "util(1,1) = (sum(Pci.*Cmatrix)+Pvector)*wei;"
        # "util(2,1) = ... +Pvector ..."
        # Different signs!
        
        term1 = np.sum(Pci * Cmatrix, axis=0) + Pvector
        
        if u_norm == 'std':
            util[0] = np.dot(term1, wei)
            util[1] = util[0] / (-np.sum(Pc * np.log2(Pc + eps)))
        else:
            util[0] = np.dot(term1, wei / Pvector)
            util[1] = np.dot(term1, wei / np.sqrt(Pvector)) / np.sqrt(-np.sum(Pc * np.log2(Pc + eps)))
            
    elif u_type == 'u_cos':
        for i in range(r):
            start = int(sumKi[i])
            end = int(sumKi[i+1])
            tmp = C[:, start:end]
            Cmatrix[:, i] = np.sqrt(np.sum(tmp**2, axis=1))
            
        term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
        
        if u_norm == 'std':
            util[0] = np.dot(term1, wei)
            util[1] = util[0] / np.sqrt(np.sum(Pc**2))
        else:
            util[0] = np.dot(term1, wei / Pvector)
            util[1] = np.dot(term1, wei / np.sqrt(Pvector)) / np.sqrt(np.sqrt(np.sum(Pc**2)))
            
    elif u_type == 'u_lp':
        p = U[2]
        for i in range(r):
            start = int(sumKi[i])
            end = int(sumKi[i+1])
            tmp = C[:, start:end]
            Cmatrix[:, i] = np.sum(tmp**p, axis=1)**(1/p)
            
        term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
        
        if u_norm == 'std':
            util[0] = np.dot(term1, wei)
            util[1] = util[0] / (np.sum(Pc**p)**(1/p))
        else:
            util[0] = np.dot(term1, wei / Pvector)
            util[1] = np.dot(term1, wei / np.sqrt(Pvector)) / np.sqrt(np.sum(Pc**p)**(1/p))
            
    return util
