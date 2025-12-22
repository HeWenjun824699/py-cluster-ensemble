import numpy as np

def relabel_cl(E):
    """
    Relabels clusters in the ensemble E.
    
    Parameters:
    E (numpy.ndarray): N-by-M matrix of cluster ensemble
    
    Returns:
    tuple: (newE, no_allcl)
        newE (numpy.ndarray): N-by-M matrix of relabeled ensemble
        no_allcl (int): total number of clusters in the ensemble
    """
    N, M = E.shape
    newE = np.zeros((N, M), dtype=int)

    # First clustering
    ucl = np.unique(E[:, 0])
    # Check if relabeling is needed (simplified from MATLAB logic, just always relabel for consistency)
    for j, val in enumerate(ucl):
        newE[E[:, 0] == val, 0] = j + 1  # 1-based indexing to match MATLAB logic implicitly if needed, or just distinct

    # The rest of the clustering
    for i in range(1, M):
        ucl = np.unique(E[:, i])
        prev_cl = len(np.unique(newE[:, :i])) # Count unique labels in previous columns
        for j, val in enumerate(ucl):
            newE[E[:, i] == val, i] = prev_cl + j + 1

    no_allcl = np.max(newE)
    return newE, no_allcl
