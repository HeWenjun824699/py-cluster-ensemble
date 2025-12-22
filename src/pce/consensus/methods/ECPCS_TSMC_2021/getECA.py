import numpy as np

def getECA(bcs, Sim):
    """
    Compute the enhanced co-association (ECA) matrix.
    
    Corresponds to getECA.m
    
    Parameters:
    -----------
    bcs : numpy.ndarray
        (N, M) matrix of globally unique cluster labels.
    Sim : numpy.ndarray
        (TotalClusters, TotalClusters) similarity matrix.
        
    Returns:
    --------
    ECA : numpy.ndarray
        (N, N) enhanced co-association matrix.
    """
    N, M = bcs.shape
    ECA = np.zeros((N, N))
    
    # Sim = Sim-diag(diag(Sim))+diag(ones(size(Sim,1),1));
    # Effectively sets diagonal to 1
    Sim = Sim.copy()
    np.fill_diagonal(Sim, 1.0)
    
    for m in range(M):
        # ECA = ECA + Sim(bcs(:,m),bcs(:,m));
        # bcs(:, m) are the cluster indices for all objects in m-th clustering
        indices = bcs[:, m]
        
        # We extract the submatrix corresponding to these indices from Sim.
        # This creates an N x N matrix where element (i, j) is Sim(cluster_of_i, cluster_of_j)
        # Python advanced indexing: Sim[rows][:, cols]
        
        # Construct the submatrix efficiently
        # Sim[indices, :] selects rows corresponding to object cluster IDs
        # [:, indices] selects columns
        sub_sim = Sim[indices][:, indices]
        
        ECA += sub_sim
        
    ECA = ECA / M
    return ECA
