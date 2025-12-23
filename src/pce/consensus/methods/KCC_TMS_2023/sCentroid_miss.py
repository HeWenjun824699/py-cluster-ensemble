import numpy as np

def sCentroid_miss(idx, K, r, Ki, sumKi):
    """
    Initialize centroid for missing data.
    idx: (K, r) - -1 for missing.
    Ki: (r,)
    sumKi: (r+1,)
    """
    total_cols = int(sumKi[-1])
    C = np.zeros((K, total_cols))
    
    for l in range(K):
        for i in range(r):
            if idx[l, i] >= 0: # Not missing (assume -1 is missing)
                col_idx = int(idx[l, i] + sumKi[i])
                C[l, col_idx] = 1
            else:
                # Random sample from 0..Ki[i]-1
                rand_c = np.random.randint(Ki[i])
                col_idx = int(rand_c + sumKi[i])
                C[l, col_idx] = 1
                
    return C
