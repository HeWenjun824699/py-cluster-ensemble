import numpy as np
import time
from .sCentroid import sCentroid
from .gCentroid import gCentroid
from .UCompute import UCompute
from .sCentroid_miss import sCentroid_miss
from .gCentroid_miss import gCentroid_miss
from .UCompute_miss import UCompute_miss

def KCC(IDX, K, U, w, weight, distance, maxIter, minThres, utilFlag, missFlag, missMatrix, n, r, Ki, sumKi, binIDX, Pvector):
    """
    KCC Core Logic.
    """
    # Initialize centroids
    # randsample(n, K)
    rand_indices = np.random.choice(n, K, replace=False)
    
    sumbest = np.inf
    converge = np.full(100, -1.0)
    utility = None
    if utilFlag == 1:
        utility = np.full((100, 2), -1.0)
        
    C = None
    if missFlag == 1:
        C = sCentroid_miss(IDX[rand_indices, :], K, r, Ki, sumKi)
    else:
        C = sCentroid(IDX[rand_indices, :], K, r, sumKi)
        
    for i in range(maxIter):
        # D = feval(distance, ...)
        if missFlag == 1:
            D = distance(U, C, weight, n, r, K, sumKi, binIDX, missMatrix)
        else:
            D = distance(U, C, weight, n, r, K, sumKi, binIDX)
            
        # [d, idx] = min(D, [], 2)
        idx = np.argmin(D, axis=1)
        d = np.min(D, axis=1)
        totalsum = np.sum(d)
        
        if abs(sumbest - totalsum) < minThres:
            break
        elif totalsum < sumbest:
            index = idx
            if missFlag == 1:
                C = gCentroid_miss(IDX, index, K, n, r, sumKi, Ki)
            else:
                C = gCentroid(IDX, index, K, n, r, sumKi, Ki)
                
            sumbest = totalsum
            if i < 100:
                converge[i] = sumbest
                if utilFlag == 1:
                    if missFlag == 1:
                        utility[i, :] = UCompute_miss(index, U, w, C, n, r, K, sumKi, Pvector, missMatrix)
                    else:
                        utility[i, :] = UCompute(index, U, w, C, n, r, K, sumKi, Pvector)
        else:
            # Objective increased
            # warning...
            break
            
    return sumbest, index, converge, utility
