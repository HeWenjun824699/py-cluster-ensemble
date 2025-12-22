import numpy as np
from getAllSegs import getAllSegs
from computeECI import computeECI
from runLWGP import runLWGP

def LWGP_v1(baseCls, nCluster, para_theta=0.4):
    """
    Locally Weighted Graph Partitioning (LWGP).
    
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
        
    # Get all clusters in the ensemble
    bcs, baseClsSegs = getAllSegs(baseCls)
    
    # Compute ECI
    ECI = computeECI(bcs, baseClsSegs, para_theta)
    
    # Perform LWGP
    label = runLWGP(bcs, baseClsSegs, ECI, ks)
    
    return label
