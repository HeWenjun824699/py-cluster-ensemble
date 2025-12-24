import numpy as np
from scipy import sparse

def EuDist2(fea_a, fea_b=None, bSqrt=True):
    """
    Euclidean Distance Matrix calculation.
    
    fea_a: nSample_a * nFeature
    fea_b: nSample_b * nFeature (optional)
    bSqrt: bool, whether to take the square root (default True)
    """
    
    if fea_b is None:
        if sparse.issparse(fea_a):
            aa = fea_a.power(2).sum(axis=1)
            ab = fea_a @ fea_a.T
            aa = np.asarray(aa) # Ensure dense for broadcasting
        else:
            aa = np.sum(fea_a * fea_a, axis=1, keepdims=True)
            ab = fea_a @ fea_a.T

        # D = aa + aa' - 2*ab
        D = aa + aa.T - 2 * ab
        
        # Numerical stability
        D[D < 0] = 0
        
        if bSqrt:
            D = np.sqrt(D)
        
        # Ensure symmetry
        D = np.maximum(D, D.T)
        
    else:
        if sparse.issparse(fea_a):
            aa = fea_a.power(2).sum(axis=1)
            if sparse.issparse(fea_b):
                 bb = fea_b.power(2).sum(axis=1)
            else:
                 bb = np.sum(fea_b * fea_b, axis=1, keepdims=True)
            ab = fea_a @ fea_b.T
            aa = np.asarray(aa)
            bb = np.asarray(bb)
        else:
            aa = np.sum(fea_a * fea_a, axis=1, keepdims=True)
            bb = np.sum(fea_b * fea_b, axis=1, keepdims=True)
            ab = fea_a @ fea_b.T

        # D = aa + bb' - 2*ab
        D = aa + bb.T - 2 * ab
        
        D[D < 0] = 0
        
        if bSqrt:
            D = np.sqrt(D)
            
    return D
