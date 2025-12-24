import numpy as np
from scipy import sparse

def NormalizeFea(fea, row=1):
    """
    Normalize features to have unit norm.
    
    fea: data matrix
    row: 1 for row normalization, 0 for column normalization
    """
    if row:
        nSmp = fea.shape[0]
        if sparse.issparse(fea):
            # Sparse case
            # sum(fea.^2, 2)
            feaNorm = fea.power(2).sum(axis=1)
            feaNorm = np.maximum(1e-14, feaNorm)
            # spdiags(feaNorm.^-.5, ...) * fea
            # In python, multiplying by diagonal matrix is efficiently done via broadcasting or multiplication
            norm_inv = np.power(feaNorm, -0.5)
            # Create diagonal matrix
            D = sparse.spdiags(norm_inv.flatten(), 0, nSmp, nSmp)
            fea = D @ fea
        else:
            # Dense case
            feaNorm = np.sum(fea**2, axis=1)
            feaNorm = np.maximum(1e-14, feaNorm)
            norm_inv = np.power(feaNorm, -0.5)
            fea = fea * norm_inv[:, np.newaxis]
            
    else:
        nSmp = fea.shape[1]
        if sparse.issparse(fea):
            feaNorm = fea.power(2).sum(axis=0)
            feaNorm = np.maximum(1e-14, feaNorm)
            norm_inv = np.power(feaNorm, -0.5)
            D = sparse.spdiags(norm_inv.flatten(), 0, nSmp, nSmp)
            fea = fea @ D
        else:
            feaNorm = np.sum(fea**2, axis=0)
            feaNorm = np.maximum(1e-14, feaNorm)
            norm_inv = np.power(feaNorm, -0.5)
            fea = fea * norm_inv[np.newaxis, :]
            
    return fea
