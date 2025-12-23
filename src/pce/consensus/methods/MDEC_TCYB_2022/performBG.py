import numpy as np
from .Tcut import Tcut

def performBG(baseClsSegs, ECI, clsNum):
    """
    Corresponds to performBG.m
    
    Args:
        baseClsSegs: (nCls, N)
        ECI: (nCls,) or (nCls, 1)
        clsNum: Number of clusters
    """
    # weightedB = bsxfun(@times, baseClsSegs, ECI); 
    # ECI broadcast to match baseClsSegs
    # baseClsSegs is (nCls, N)
    
    eci_flat = np.array(ECI).flatten()
    weightedB = baseClsSegs * eci_flat[:, np.newaxis]
    
    # label = Tcut(weightedB',clsNum);
    # weightedB' -> (N, nCls)
    label = Tcut(weightedB.T, clsNum)
    
    return label
