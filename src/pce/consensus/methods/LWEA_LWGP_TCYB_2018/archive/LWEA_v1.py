import numpy as np
from getAllSegs import getAllSegs
from computeECI import computeECI
from computeLWCA import computeLWCA
from runLWEA import runLWEA

def LWEA_v1(baseCls, nCluster, para_theta=0.4):
    """
    Locally Weighted Ensemble Clustering (LWEA).
    
    Args:
        baseCls: N x M matrix of base clusterings.
        nCluster: Number of clusters (int) or list of numbers of clusters.
        para_theta: Parameter theta (default 0.4).
        
    Returns:
        label: Clustering results.
    """
    # Ensure nCluster is iterable if scalar
    if np.isscalar(nCluster):
        ks = [nCluster]
    else:
        ks = nCluster
        
    mBase = baseCls.shape[1]
    
    # Get all clusters in the ensemble
    bcs, baseClsSegs = getAllSegs(baseCls)
    
    # Compute ECI
    ECI = computeECI(bcs, baseClsSegs, para_theta)
    
    # Compute LWCA
    LWCA = computeLWCA(baseClsSegs, ECI, mBase)
    
    # Run LWEA (HAC)
    label = runLWEA(LWCA, ks)
    
    return label
