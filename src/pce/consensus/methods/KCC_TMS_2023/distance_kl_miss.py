import numpy as np

def distance_kl_miss(U, C, weight, n, r, K, sumKi, binIDX, M):
    D = np.zeros((n, K))
    eps = np.finfo(float).eps
    
    for l in range(n):
        idx_mask = M[l, :].astype(bool)
        if np.any(idx_mask):
            w_idx = weight[idx_mask]
            c_indices = binIDX[l, idx_mask].astype(int)
            c_part = C[:, c_indices]
            
            D[l, :] = -np.dot(np.log2(c_part + eps), w_idx)
            
    return D
