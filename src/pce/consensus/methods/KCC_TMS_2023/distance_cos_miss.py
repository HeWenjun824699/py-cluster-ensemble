import numpy as np

def distance_cos_miss(U, C, weight, n, r, K, sumKi, binIDX, M):
    D = np.zeros((n, K))
    D1 = np.zeros((K, r))
    
    for i in range(r):
        start = int(sumKi[i])
        end = int(sumKi[i+1])
        D1[:, i] = np.sqrt(np.sum(C[:, start:end]**2, axis=1))
        
    for l in range(n):
        idx_mask = M[l, :].astype(bool)
        if np.any(idx_mask):
            w_idx = weight[idx_mask]
            d1_idx = D1[:, idx_mask]
            c_indices = binIDX[l, idx_mask].astype(int)
            c_part = C[:, c_indices]
            
            term = np.dot(c_part / (d1_idx + 1e-10), w_idx)
            D[l, :] = np.sum(w_idx) - term
            
    return D
