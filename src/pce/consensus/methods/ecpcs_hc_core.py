import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from .ECPCS_TSMC_2021.getAllSegs import getAllSegs
from .ECPCS_TSMC_2021.simxjac import simxjac
from .ECPCS_TSMC_2021.computePTS_II import computePTS_II
from .ECPCS_TSMC_2021.getECA import getECA

def ecpcs_hc_core(baseCls, t, K):
    """
    ECPCS-HC: Enhanced Ensemble Clustering via Fast Propagation of Cluster-wise Similarities.
    
    Corresponds to ECPCS_HC.m
    
    Parameters:
    -----------
    baseCls : numpy.ndarray
        (N, nBC) matrix of base cluster labels.
    t : int
        Number of steps for random walk (e.g., 20).
    K : int or list of int
        Number of clusters desired.
        
    Returns:
    --------
    Label : numpy.ndarray
        (N, len(K)) matrix of cluster labels.
    """
    # 1. Get segments (clusters) and refined base clustering labels
    # [bcs, baseClsSegs] = getAllSegs(baseCls);
    bcs, baseClsSegs = getAllSegs(baseCls)
    
    # 2. Build cluster similarity matrix by Jaccard coefficient
    # clsSim = full(simxjac(baseClsSegs));
    clsSim = simxjac(baseClsSegs)
    
    # 3. Perform random walks and obtain a new cluster-wise similarity matrix
    # clsSimRW = computePTS_II(clsSim, t);
    clsSimRW = computePTS_II(clsSim, t)
    
    # 4. Get Enhanced Co-Association Matrix
    # ECA = getECA(bcs,clsSimRW);
    ECA = getECA(bcs, clsSimRW)
    
    # 5. Run Hierarchical Clustering
    # Label = runHC(ECA, K);
    Label = runHC(ECA, K)
    
    return Label

def runHC(S, ks):
    """
    Hierarchical Clustering on Similarity Matrix.
    
    Corresponds to local function runHC inside ECPCS_HC.m
    """
    # S is similarity. Convert to distance.
    # d = stod(S); in Matlab converts Similarity matrix to Distance vector for linkage
    
    # Ensure S has 1s on diagonal (distance 0)
    S_mod = S.copy()
    np.fill_diagonal(S_mod, 1.0)
    
    # Distance matrix = 1 - Similarity
    D = 1.0 - S_mod
    D[D < 0] = 0 # Numerical stability
    
    # Squareform converts full distance matrix to condensed distance vector required by linkage
    # checks=False allows for small numerical deviations from symmetry
    d_vec = squareform(D, checks=False)
    
    # Zal = linkage(d,'average');
    Zal = linkage(d_vec, method='average')
    
    if np.isscalar(ks):
        ks_list = [ks]
    else:
        ks_list = ks
        
    N = S.shape[0]
    Label = np.zeros((N, len(ks_list)), dtype=int)
    
    for i, k in enumerate(ks_list):
        # Label(:,iK) = cluster(Zal,'maxclust',ks(iK));
        lab = fcluster(Zal, k, criterion='maxclust')
        Label[:, i] = lab
        
    return Label