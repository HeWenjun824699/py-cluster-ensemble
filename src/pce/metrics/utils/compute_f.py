import numpy as np


def compute_f(T, H):
    """
    Compute F-score, Precision, Recall.
    T: Ground truth labels (vector)
    H: Computed/Hypothesis labels (vector)
    """
    T = np.array(T).flatten()
    H = np.array(H).flatten()
    
    if len(T) != len(H):
        print('Size T:', T.shape)
        print('Size H:', H.shape)
        # In MATLAB logic, it might error or continue, here we let it assume valid input or fail.
        
    N = len(T)
    numT = 0
    numH = 0
    numI = 0
    
    # O(N^2) loop as per MATLAB code
    for n in range(N):
        # T(n+1:end) in MATLAB corresponds to T[n+1:] in Python
        Tn = (T[n+1:] == T[n])
        Hn = (H[n+1:] == H[n])
        
        numT += np.sum(Tn)
        numH += np.sum(Hn)
        numI += np.sum(Tn * Hn) # Element-wise AND/Product
        
    p = 1.0
    r = 1.0
    f = 1.0
    
    if numH > 0:
        p = numI / numH
    
    if numT > 0:
        r = numI / numT
        
    if (p + r) == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)
        
    return f, p, r
