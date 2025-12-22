import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def runLWEA(S, ks):
    """
    Run LWEA algorithm (Average Linkage on Similarity Matrix).
    
    Args:
        S: N x N Similarity matrix (Co-association matrix).
        ks: List or array of cluster numbers to obtain.
        
    Returns:
        resultsLWEA: N x len(ks) matrix of cluster labels.
    """
    N = S.shape[0]
    
    # Convert similarity matrix to distance vector
    # Matlab: d = stod2(S) -> d = 1 - s (condensed)
    # scipy squareform(S) returns condensed.
    # But we want distance = 1 - similarity.
    # We can do squareform(1 - S).
    # We must ensure the diagonal is 0 for distance (1 for similarity).
    # The previous step in computeLWCA sets diagonal to 1.
    
    # Ensure S is 1 on diagonal (already done in computeLWCA, but good for safety)
    # and symmetric.
    
    # Distance matrix (dense)
    D_mat = 1.0 - S
    np.fill_diagonal(D_mat, 0.0) # Distance to self is 0
    
    # Convert to condensed distance vector for linkage
    d = squareform(D_mat, checks=False)
    
    # Average linkage
    Zal = linkage(d, method='average')
    
    resultsLWEA = np.zeros((N, len(ks)), dtype=int)
    
    for i, K in enumerate(ks):
        # cluster(Zal, 'maxclust', K)
        # fcluster returns 1-based cluster labels
        labels = fcluster(Zal, K, criterion='maxclust')
        resultsLWEA[:, i] = labels
        
    return resultsLWEA

def stod2(S):
    """
    Matlab equivalent helper (logic embedded in runLWEA above).
    Provided for reference.
    """
    # In Python/SciPy, squareform handles this.
    pass
