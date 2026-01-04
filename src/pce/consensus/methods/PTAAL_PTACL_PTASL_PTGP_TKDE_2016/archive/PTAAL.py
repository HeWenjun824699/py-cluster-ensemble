import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from computeMicroclusters import compute_microclusters
from computeMCA import compute_mca
from computePTS_fast_v3 import compute_pts_fast_v3
from mapMicroclustersBackToObjects import map_microclusters_back_to_objects

def pta_al(base_cls, n_cluster):
    """
    Produce microclusters and run PTA-AL.
    
    Args:
        base_cls: N x M matrix of cluster labels.
        n_cluster: Number of clusters (scalar or list).
        
    Returns:
        label_al: Clustering results (N x numel(n_cluster)).
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
    
    # Run PTA-AL
    if np.isscalar(n_cluster):
        ks = [n_cluster]
    else:
        ks = n_cluster
        
    mc_results_al = run_pta_al(pts, ks)
    
    label_al = map_microclusters_back_to_objects(mc_results_al, mc_labels)
    
    return label_al

def run_pta_al(S, ks):
    """
    Run PTA-AL (Average Linkage).
    
    Args:
        S: Similarity matrix.
        ks: List of cluster numbers.
        
    Returns:
        results_al: Clustering results (N x len(ks)).
    """
    n = S.shape[0]
    
    # Convert similarity matrix to distance vector
    # STRICTLY match Matlab's stod2 logic using explicit loops to ensure exact index order.
    # Matlab:
    # nextIdx = 1;
    # for a = 1:N-1
    #    s(nextIdx:nextIdx+(N-a-1)) = S(a,[a+1:end]);
    #    nextIdx = nextIdx + N - a;
    # end
    # d = 1 - s;
    
    d_list = []
    for i in range(n - 1):
        # S[i, i+1:] corresponds to S(a, a+1:end) in Matlab (1-based a=i+1)
        d_list.append(1.0 - S[i, i+1:])
        
    d_condensed = np.concatenate(d_list)
    
    # Average Linkage
    zal = linkage(d_condensed, method='average')
    
    results_al = np.zeros((n, len(ks)), dtype=int)
    
    for i, k in enumerate(ks):
        # fcluster returns 1-based cluster labels
        results_al[:, i] = fcluster(zal, k, criterion='maxclust')
        
    return results_al
