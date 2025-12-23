import numpy as np

def distance_kl(U, C, weight, n, r, K, sumKi, binIDX):
    """
    KL Divergence distance.
    """
    D = np.zeros((n, K))
    eps = np.finfo(float).eps
    
    for l in range(n):
        c_part = C[:, binIDX[l, :].astype(int)] # (K, r)
        # -log2(c_part + eps) * weight
        D[l, :] = -np.dot(np.log2(c_part + eps), weight)
        
    return D
