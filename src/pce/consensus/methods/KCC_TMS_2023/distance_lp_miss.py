import numpy as np

def distance_lp_miss(U, C, weight, n, r, K, sumKi, binIDX, M):
    p = U[2]
    D = np.zeros((n, K))
    D1 = np.zeros((K, r))
    
    for i in range(r):
        start = int(sumKi[i])
        end = int(sumKi[i+1])
        D1[:, i] = np.sum(C[:, start:end]**p, axis=1)**(1/p)
        
    for l in range(n):
        idx_mask = M[l, :].astype(bool)
        if np.any(idx_mask):
            w_idx = weight[idx_mask]
            d1_idx = D1[:, idx_mask]
            c_indices = binIDX[l, idx_mask].astype(int)
            c_part = C[:, c_indices]
            
            ratio = c_part / (d1_idx + 1e-10)
            term = np.dot(ratio**(p-1), w_idx)
            D[l, :] = np.sum(w_idx) - term
            
    return D
