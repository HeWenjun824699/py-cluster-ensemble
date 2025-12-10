import numpy as np
from scipy.special import comb
from .contingency import contingency


def rand_index(c1, c2):
    """
    Calculates Rand Indices to compare two partitions.
    
    Returns:
    AR : Adjusted Rand Index
    RI : Unadjusted Rand Index
    MI : Mirkin's Index
    HI : Hubert's Index
    """
    c1 = np.array(c1)
    c2 = np.array(c2)
    
    if c1.ndim > 1 or c2.ndim > 1:
        # Flatten if they are not 1D, though logic expects vectors
        c1 = c1.flatten()
        c2 = c2.flatten()

    C = contingency(c1, c2)
    
    n = np.sum(C)
    nis = np.sum(np.sum(C, axis=1)**2) # sum of squares of sums of rows
    njs = np.sum(np.sum(C, axis=0)**2) # sum of squares of sums of columns
    
    t1 = comb(n, 2)
    t2 = np.sum(C**2)
    t3 = 0.5 * (nis + njs)
    
    # Expected index (for adjustment)
    nc = (n * (n**2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    
    A = t1 + t2 - t3
    D = -t2 + t3
    
    if t1 == nc:
        AR = 0
    else:
        AR = (A - nc) / (t1 - nc)
        
    RI = A / t1
    MI = D / t1
    HI = (A - D) / t1
    
    return AR, RI, MI, HI
