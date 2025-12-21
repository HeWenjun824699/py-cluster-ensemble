import numpy as np
from computeMicroclusters import compute_microclusters
from computeMCA import compute_mca
from computePTS_fast_v3 import compute_pts_fast_v3
from runPTGP_v2 import run_ptgp_v2
from mapMicroclustersBackToObjects import map_microclusters_back_to_objects

def ptgp(base_cls, n_cluster):
    """
    Produce microclusters and run PTGP.
    
    Args:
        base_cls: N x M matrix of cluster labels.
        n_cluster: Number of clusters (scalar or list).
        
    Returns:
        label_ptgp: Clustering results (N x numel(n_cluster)).
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
    
    # Perform PTGP
    if np.isscalar(n_cluster):
        ks = [n_cluster]
    else:
        ks = n_cluster
        
    mc_results_ptgp = run_ptgp_v2(mc_base_cls, pts, ks)
    
    label_ptgp = map_microclusters_back_to_objects(mc_results_ptgp, mc_labels)
    
    return label_ptgp
