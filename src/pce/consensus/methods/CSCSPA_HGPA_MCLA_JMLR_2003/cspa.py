import numpy as np
from clstoclbs import clstoclbs
from checks import checks
from metis import metis

def cspa(cls, k=None):
    """
    Cluster-based Similarity Partitioning Algorithm (CSPA)
    
    Parameters:
    cls (numpy.ndarray): Matrix of cluster labels (n_clusterings x n_samples)
    k (int): Desired number of clusters. If None, defaults to max label in cls.
    
    Returns:
    numpy.ndarray: Consensus cluster labels (1D array)
    """
    
    if k is None:
        k = int(np.max(cls))
        # Note: If labels are 0-based (0..k-1), max is k-1.
        # If we interpret k as 'number of clusters', and max(cls) is e.g. 2 (labels 0,1,2),
        # then we might want k=3. 
        # However, following MATLAB logic exactly: k = max(max(cls)).
        # It's recommended to pass k explicitly.

    clbs = clstoclbs(cls)
    
    # Calculate similarity matrix (Binary cluster sharing)
    s = clbs.T @ clbs
    
    # Normalize and check
    s = checks(s / cls.shape[0])
    
    # Partition
    cl = metis(s, k)
    
    return cl
