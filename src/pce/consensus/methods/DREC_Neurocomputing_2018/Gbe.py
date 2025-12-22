import numpy as np
from .relabelCl import relabel_cl

def gbe(E):
    """
    Generate Binary labeling matrix E.
    
    Parameters:
    E (numpy.ndarray): n*M matrix of clustering results
    
    Returns:
    numpy.ndarray: BE: n*KM matrix
    """
    E_relabeled, KM = relabel_cl(E)
    N = E_relabeled.shape[0]
    
    # BE = zeros(N,KM)
    # Python 0-based index vs relabel_cl 1-based index (if preserved from MATLAB logic)
    # In relabel_cl.py, I assigned j+1, so labels are 1...KM.
    # So we need KM columns. Indexing will be val-1.
    
    BE = np.zeros((N, KM))
    
    # Vectorized assignment
    # Create row indices corresponding to the flattened E_relabeled
    row_indices = np.repeat(np.arange(N), E_relabeled.shape[1])
    col_indices = E_relabeled.flatten() - 1
    
    BE[row_indices, col_indices] = 1
        
    return BE
