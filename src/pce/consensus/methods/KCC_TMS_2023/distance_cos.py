import numpy as np

def distance_cos(U, C, weight, n, r, K, sumKi, binIDX):
    D = np.zeros((n, K))
    D1 = np.zeros((K, r))
    
    for i in range(r):
        start = int(sumKi[i])
        end = int(sumKi[i+1])
        D1[:, i] = np.sqrt(np.sum(C[:, start:end]**2, axis=1))
        
    sum_weight = np.sum(weight)
    
    for l in range(n):
        c_part = C[:, binIDX[l, :].astype(int)] # (K, r)
        # sum(weight) - (c_part ./ D1) * weight
        
        term = (c_part / (D1 + 1e-10)) @ weight # Added small epsilon to avoid div by zero
        D[l, :] = sum_weight - term
        
    return D
