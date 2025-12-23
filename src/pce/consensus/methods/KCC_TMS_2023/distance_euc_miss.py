import numpy as np

def distance_euc_miss(U, C, weight, n, r, K, sumKi, binIDX, M):
    """
    M: (n, r) boolean/binary
    """
    D = np.zeros((n, K))
    D1 = np.zeros((K, r))
    
    for i in range(r):
        start = int(sumKi[i])
        end = int(sumKi[i+1])
        D1[:, i] = np.sum(C[:, start:end]**2, axis=1)
        
    for l in range(n):
        idx_mask = M[l, :].astype(bool) # (r,)
        if np.any(idx_mask):
            w_idx = weight[idx_mask]
            d1_idx = D1[:, idx_mask]
            # binIDX[l, idx_mask] gives indices for present partitions
            c_indices = binIDX[l, idx_mask].astype(int)
            c_part = C[:, c_indices]
            
            # sum(weight(idx)) + (d1_idx - 2*c_part) * weight(idx)
            term = np.dot((d1_idx - 2*c_part), w_idx)
            D[l, :] = np.sum(w_idx) + term
        else:
            # Handle all missing? Should not happen usually
            pass
            
    return D
