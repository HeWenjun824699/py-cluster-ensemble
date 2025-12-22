import numpy as np
from scipy import sparse

def discretisationEigenVectorData(EigenVector):
    """
    Discretizes previously rotated eigenvectors.
    
    Corresponds to discretisationEigenVectorData.m
    
    Parameters:
    -----------
    EigenVector : numpy.ndarray
        (n, k) matrix.
        
    Returns:
    --------
    Y : scipy.sparse.csr_matrix or numpy.ndarray
        (n, k) indicator matrix.
    """
    n, k = EigenVector.shape
    
    # [Maximum,J]=max(EigenVector');
    # In Matlab, max(A') finds max along columns of transpose, i.e., max along rows of A.
    # J is the index of the max value in each row.
    J = np.argmax(EigenVector, axis=1)
    
    # Y=sparse(1:n,J',1,n,k);
    # Construct sparse matrix
    # Rows: 0 to n-1
    # Cols: J
    # Values: 1
    
    row_indices = np.arange(n)
    col_indices = J
    data = np.ones(n)
    
    Y = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, k))
    
    return Y
