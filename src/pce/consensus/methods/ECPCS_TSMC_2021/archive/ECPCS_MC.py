import numpy as np
from pce.consensus.methods.ECPCS_TSMC_2021.getAllSegs import getAllSegs
from pce.consensus.methods.ECPCS_TSMC_2021.simxjac import simxjac
from pce.consensus.methods.ECPCS_TSMC_2021.computePTS_II import computePTS_II
from pce.consensus.methods.ECPCS_TSMC_2021.Ncut.ncutW_2 import ncutW_2

def ECPCS_MC(baseCls, t, K):
    """
    ECPCS-MC: Enhanced Ensemble Clustering via Fast Propagation of Cluster-wise Similarities (Meta-Clustering).
    
    Corresponds to ECPCS_MC.m
    
    Parameters:
    -----------
    baseCls : numpy.ndarray
        (N, M) matrix of base cluster labels.
    t : int
        Number of steps for random walk.
    K : int
        Number of clusters.
        
    Returns:
    --------
    Label : numpy.ndarray
        (N,) vector of cluster labels.
    """
    baseCls = np.array(baseCls)
    N, M = baseCls.shape
    
    # [bcs, baseClsSegs] = getAllSegs(baseCls);
    bcs, baseClsSegs = getAllSegs(baseCls)
    
    # clsSim = full(simxjac(baseClsSegs));
    clsSim = simxjac(baseClsSegs)
    
    # clsSimRW = computePTS_II(clsSim, t);
    clsSimRW = computePTS_II(clsSim, t)
    
    # Meta-clustering
    # clsL = ncutW_2(clsSimRW, K);
    clsL = ncutW_2(clsSimRW, K)
    
    # for j = 2:size(clsL,2)
    #    clsL(:,j) = clsL(:,j) * j;
    # end
    # clsL is (nClusters, K)
    for j in range(1, clsL.shape[1]): # j from 1 to K-1 (indices)
        # Matlab j starts at 2, so column index 1.
        # Matlab: clsL(:,j) * j. If j=2 (2nd col), multiply by 2.
        # Python: col index j (0-based). 
        # Logic: We want to convert one-hot to labels 1..K.
        # Col 0 -> 1 (implicit multiplier 1)
        # Col 1 -> 2
        # ...
        # Col j -> j+1
        clsL[:, j] = clsL[:, j] * (j + 1)
        
    # Col 0 is multiplied by 1 implicitly (no change needed if it's 0/1)
    # But wait, if it's 0/1, col 0 is 1 where cluster 1 is.
    # So we just sum them up.
    
    # clsLabel = sum(clsL');
    # Matlab sum(A) sums columns. sum(clsL') sums rows of clsL.
    clsLabel = np.sum(clsL, axis=1)
    
    # clsLabel_cum = zeros(K, N);
    clsLabel_cum = np.zeros((K, N))
    
    # for i=1:max(clsLabel),
    #    matched_clusters = find(clsLabel==i);
    #    clsLabel_cum(i,:) = mean(baseClsSegs(matched_clusters,:),1);
    # end;
    
    max_label = int(np.max(clsLabel))
    # Note: clsLabel indices are 1-based (1..K) because we multiplied by j=2..K and col 1 (implied 1).
    # Ideally clsL is a valid indicator matrix where each row has exactly one 1.
    
    for i in range(1, max_label + 1):
        # matched_clusters = find(clsLabel==i);
        # Python indices
        matched_clusters = np.where(clsLabel == i)[0]
        
        if len(matched_clusters) > 0:
            # clsLabel_cum(i,:) = mean(baseClsSegs(matched_clusters,:),1);
            # baseClsSegs is (TotalClusters, N)
            # mean over matched rows
            # Assign to row i-1 in Python (0-based)
            clsLabel_cum[i-1, :] = np.mean(baseClsSegs[matched_clusters, :], axis=0)
            
    # [~,Label]=max(clsLabel_cum);
    # Max along columns (for each object, which meta-cluster is best)
    # Matlab max(A) -> max of columns.
    # Label is row index.
    Label = np.argmax(clsLabel_cum, axis=0)
    
    # Convert to 1-based labels to match typical Matlab output if desired, 
    # but usually Python uses 0-based. The code returns indices.
    # Let's keep 0-based or 1-based?
    # Matlab returns 1..K. Python argmax returns 0..K-1.
    # If the user expects consistency with Matlab result values, +1.
    # But often Python users expect 0-based.
    # However, for "strict consistency", if Matlab returns 1,2,3...
    # I should probably return 1,2,3... or just document it.
    # Given other functions return labels, usually 0-based is standard in sklearn.
    # But this is a port.
    # Let's stick to Python standard 0-based for the return, 
    # as `getAllSegs` remapped inputs to 0-based internal logic anyway.
    # But wait, `getAllSegs` in Matlab returned whatever baseCls had? 
    # No, `getAllSegs` in Python remapped to 0..k-1.
    # So the whole pipeline is internally consistent.
    # I will return 0-based labels.
    
    return Label
