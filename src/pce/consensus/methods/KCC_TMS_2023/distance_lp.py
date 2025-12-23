import numpy as np

def distance_lp(U, C, weight, n, r, K, sumKi, binIDX):
    p = U[2]
    D = np.zeros((n, K))
    D1 = np.zeros((K, r))
    
    for i in range(r):
        start = int(sumKi[i])
        end = int(sumKi[i+1])
        D1[:, i] = np.sum(C[:, start:end]**p, axis=1)**(1/p)
        
    sum_weight = np.sum(weight)
    
    for l in range(n):
        c_part = C[:, binIDX[l, :].astype(int)]
        # sum(weight) - ((c_part ./ D1).^(p-1)) * weight
        
        ratio = c_part / (D1 + 1e-10)
        term = np.dot(ratio**(p-1), weight)
        D[l, :] = sum_weight - term
        
    return D
