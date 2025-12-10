import numpy as np
from .clstoclbs import clstoclbs
from .clcgraph import clcgraph
from .clbtocl import clbtocl


def mcla(cls, k=None):
    """
    Performs MCLA for CLUSTER ENSEMBLES.
    
    MATLAB equivalent:
    function cl = mcla(cls,k)
    """
    
    # cls: (n_samples, n_clusterings) or (n_clusterings, n_samples)?
    # Assuming convention: Rows=Samples, Cols=Clusterings is NOT standard in some MATLAB code.
    # Let's check clstoclbs.m again.
    # clstoclbs calls cltoclb(cls(i,:)).
    # If cls(i,:) is a clustering, then cls has Rows=Clusterings.
    # This implies cls is (n_clusterings, n_samples).
    # PLEASE NOTE: This is transposed relative to scikit-learn convention.
    # I will assume the input cls follows the MATLAB shape: (n_clusterings, n_samples).
    
    if k is None:
        k = np.max(cls) + 1 # Assuming 0-based labels in cls.
        # If 1-based, max is sufficient count.
        # If cls contains 1..K, max is K.
        # If cls contains 0..K-1, max is K-1.
        # Let's assume the user handles 'k' or we infer max number of clusters.
    
    print('mcla: preparing graph for meta-clustering')
    
    # clb = clstoclbs(cls);
    # clstoclbs should return a binary matrix of all clusters vs samples.
    clb = clstoclbs(cls)
    
    # cl_lab = clcgraph(clb,k,'simbjac');
    # Partitions the hypergraph (clusters) into k meta-clusters.
    cl_lab = clcgraph(clb, k, 'simbjac')
    
    # cl_lab contains labels for each row of clb (each cluster in the ensemble).
    # Determine number of meta-clusters found (should be k, but let's be safe).
    n_meta_clusters = np.max(cl_lab) + 1
    n_samples = clb.shape[1]
    
    clb_cum = np.zeros((n_meta_clusters, n_samples))
    
    # for i=1:max(cl_lab)
    for i in range(n_meta_clusters):
        matched_clusters = np.where(cl_lab == i)[0]
        
        if len(matched_clusters) > 0:
            # clb_cum(i,:) = mean(clb(matched_clusters,:),1);
            # Average the binary vectors of all clusters belonging to meta-cluster i
            clb_cum[i, :] = np.mean(clb[matched_clusters, :], axis=0)
            
    # cl = clbtocl(clb_cum);
    # Assign samples to meta-clusters
    cl, _ = clbtocl(clb_cum)
    
    return cl
