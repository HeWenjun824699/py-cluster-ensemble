import numpy as np

def distance_euc(U, C, weight, n, r, K, sumKi, binIDX):
    """
    Euclidean distance.
    C: (K, total_cols)
    weight: (r,)
    binIDX: (n, r)
    """
    D = np.zeros((n, K))
    D1 = np.zeros((K, r))
    
    for i in range(r):
        start = int(sumKi[i])
        end = int(sumKi[i+1])
        # sum over rows (clusters)
        D1[:, i] = np.sum(C[:, start:end]**2, axis=1)
        
    # Vectorized computation
    # D(l,:) = (sum(weight) + (D1 - 2*C(:, binIDX(l,:))) * weight)'
    
    # binIDX is (n, r). C is (K, total_cols).
    # We need C[:, binIDX[l, :]] -> (K, r)
    
    sum_weight = np.sum(weight)
    
    for l in range(n):
        # C[:, binIDX[l, :]] picks columns corresponding to binIDX[l, :]
        # Result is (K, r)
        c_part = C[:, binIDX[l, :].astype(int)]
        
        term = (D1 - 2 * c_part) @ weight # (K, r) @ (r,) -> (K,)
        D[l, :] = sum_weight + term
        
    return D
