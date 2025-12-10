import numpy as np
from .cltoclb import cltoclb
from .clhgraph import clhgraph


def hgpa(cls, k=None):
    """
    Performs HGPA (HyperGraph Partitioning Algorithm) for CLUSTER ENSEMBLES.
    
    Args:
        cls: Matrix of cluster labels (n_clusterings x n_samples)
        k: Number of desired clusters (optional, defaults to max label in cls)
        
    Returns:
        cl: Consensus cluster labels
    """
    # disp('CLUSTER ENSEMBLES using HGPA');

    if k is None:
        # Determine k as the number of unique labels
        k = len(np.unique(cls))

    r, n_samples = cls.shape
    clb_list = []
    
    for i in range(r):
        clb_list.append(cltoclb(cls[i, :]))
    
    if not clb_list:
        return np.array([])

    # clb_list contains matrices of shape (n_clusters_in_clustering_i x n_samples)
    # We stack them vertically to get a (Total Hyperedges x n_samples) matrix.
    clb = np.vstack(clb_list)
    
    # clhgraph/hmetis/wgraph(method=2) expects:
    # x: (Vertices x Hyperedges)
    # Vertices are samples. Hyperedges are the clusters from input clusterings.
    # clb is (Hyperedges x Vertices).
    # So we pass clb.T
    
    cl = clhgraph(clb.T, k)
    
    return cl
