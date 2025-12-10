import numpy as np


def contingency(mem1, mem2):
    """
    Form contingency matrix for two vectors.
    
    Parameters:
    mem1 : array-like
        Cluster assignments for first partition.
    mem2 : array-like
        Cluster assignments for second partition.
        
    Returns:
    Cont : numpy.ndarray
        Contingency matrix.
    """
    mem1 = np.array(mem1)
    mem2 = np.array(mem2)
    
    # Map labels to 0..N-1 to ensure compact matrix
    # The original MATLAB code used max(Mem) which implies integer labels > 0.
    # Here we make it robust to any label set by using unique inverse
    u1, inv1 = np.unique(mem1, return_inverse=True)
    u2, inv2 = np.unique(mem2, return_inverse=True)
    
    n1 = len(u1)
    n2 = len(u2)
    
    cont = np.zeros((n1, n2))
    
    # Equivalent to for loop: Cont(Mem1(i), Mem2(i)) += 1
    # np.add.at is unbuffered, useful for this
    # But simply:
    for i in range(len(mem1)):
        cont[inv1[i], inv2[i]] += 1
        
    return cont
