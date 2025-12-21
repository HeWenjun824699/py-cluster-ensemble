import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from computeMicroclusters import compute_microclusters
from computeMCA import compute_mca
from computePTS_fast_v3 import compute_pts_fast_v3
from mapMicroclustersBackToObjects import map_microclusters_back_to_objects

def pta_sl(base_cls, n_cluster):
    """
    Produce microclusters and run PTA-SL (Single Linkage).
    
    Args:
        base_cls: N x M matrix of cluster labels.
        n_cluster: Number of clusters (scalar or list).
        
    Returns:
        label_sl: Clustering results (N x numel(n_cluster)).
    """
    # Produce microclusters
    mc_base_cls, mc_labels = compute_microclusters(base_cls)
    tilde_n = mc_base_cls.shape[0]
    
    # Compute the microcluster based co-association matrix
    mca = compute_mca(mc_base_cls)
    
    # Set parameters K and T
    para = {}
    para['K'] = np.floor(np.sqrt(tilde_n) / 2)
    para['T'] = np.floor(np.sqrt(tilde_n) / 2)
    
    # Compute PTS
    pts = compute_pts_fast_v3(mca, mc_labels, para)
    
    # Run PTA-SL
    if np.isscalar(n_cluster):
        ks = [n_cluster]
    else:
        ks = n_cluster
        
    mc_results_sl = run_pta_sl(pts, ks)
    
    label_sl = map_microclusters_back_to_objects(mc_results_sl, mc_labels)
    
    return label_sl

def run_pta_sl(S, ks):
    """
    Run PTA-SL (Single Linkage).
    
    Args:
        S: Similarity matrix.
        ks: List of cluster numbers.
        
    Returns:
        results_sl: Clustering results (N x len(ks)).
    """
    n = S.shape[0]
    
    # Convert similarity matrix to distance vector
    # STRICTLY match Matlab's stod2 logic using explicit loops.
    
    d_list = []
    for i in range(n - 1):
        d_list.append(1.0 - S[i, i+1:])
        
    d_condensed = np.concatenate(d_list)
    
    # Single Linkage
    zsl = linkage(d_condensed, method='single')
    
    results_sl = np.zeros((n, len(ks)), dtype=int)
    
    for i, k in enumerate(ks):
        # fcluster returns 1-based cluster labels
        results_sl[:, i] = fcluster(zsl, k, criterion='maxclust')
        
    return results_sl
