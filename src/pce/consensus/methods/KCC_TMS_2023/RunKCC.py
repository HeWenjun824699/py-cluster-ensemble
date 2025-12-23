import numpy as np
import time
from .Preprocess import Preprocess
from .KCC import KCC

def RunKCC(IDX, K, U=None, w=None, rep=5, maxIter=20, minThres=1e-5, utilFlag=0):
    """
    Main Runner.
    IDX: (n, r) labels.
    """
    start_time = time.time()
    n, r = IDX.shape
    
    if U is None: U = ['u_h', 'std']
    if w is None: w = np.ones(r)
    
    Ki, sumKi, binIDX, missFlag, missMatrix, distance, Pvector, weight = \
        Preprocess(IDX, U, n, r, w, utilFlag)
        
    l_sumbest = np.zeros(rep)
    l_index = np.zeros((n, rep))
    l_converge = np.zeros((100, rep))
    l_utility = None
    
    if utilFlag == 1:
        l_utility = np.zeros((100, 2*rep))
        
    for p in range(rep):
        sumbest, index, converge, utility = KCC(
            IDX, K, U, w, weight, distance, maxIter, minThres, utilFlag, 
            missFlag, missMatrix, n, r, Ki, sumKi, binIDX, Pvector
        )
        l_sumbest[p] = sumbest
        l_index[:, p] = index
        l_converge[:, p] = converge
        if utilFlag == 1:
            l_utility[:, (2*p):(2*p+2)] = utility
            
    pos = np.argmin(l_sumbest)
    pi_sumbest = l_sumbest[pos]
    pi_index = l_index[:, pos]
    pi_converge = l_converge[:, pos]
    pi_utility = []
    if utilFlag == 1:
        pi_utility = l_utility[:, (2*pos):(2*pos+2)]
        
    t = time.time() - start_time
    
    return pi_sumbest, pi_index, pi_converge, pi_utility, t
