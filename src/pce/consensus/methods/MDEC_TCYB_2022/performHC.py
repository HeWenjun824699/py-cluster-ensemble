import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def performHC(S, clsNum):
    """
    Performs Hierarchical Clustering on similarity matrix S.
    
    Args:
        S: N-by-N similarity matrix (numpy array)
        clsNum: Number of clusters (int)
        
    Returns:
        results_al: Cluster labels (numpy array)
    """
    # Convert similarity matrix to distance vector
    # Matlab: d = stod(S); (converts 1-S upper triangle to vector)
    # Python: squareform(1-S) extracts upper triangle if input is square matrix
    
    # Ensure S is valid. Assuming S is similarity in [0, 1].
    # Distance = 1 - Similarity
    D_mat = 1.0 - S
    
    # Fill diagonal with 0 just in case
    np.fill_diagonal(D_mat, 0)
    
    # Convert to condensed distance matrix
    # squareform returns the upper triangular part as a vector (checking symmetry can be skipped)
    d = squareform(D_mat, checks=False)
    
    # Average linkage
    Zal = linkage(d, method='average')
    
    # Cut tree
    # fcluster returns 1-based labels, consistent with Matlab
    results_al = fcluster(Zal, t=clsNum, criterion='maxclust')
    
    return results_al
