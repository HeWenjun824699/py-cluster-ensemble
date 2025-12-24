import numpy as np
from scipy import sparse

def discretisationEigenVectorData(EigenVector):
    """
    Discretizes previously rotated eigenvectors.
    
    Y = discretisationEigenVectorData(EigenVector)
    """
    n, k = EigenVector.shape
    
    # [Maximum, J] = max(EigenVector'); 
    # In MATLAB max(A') finds max along columns of A', which corresponds to max along rows of A.
    # MATLAB indices are 1-based, Python 0-based.
    
    J = np.argmax(EigenVector, axis=1)
    
    # Y=sparse(1:n,J',1,n,k);
    # Construct sparse matrix. 
    # Rows: 0 to n-1
    # Cols: J
    
    rows = np.arange(n)
    cols = J
    data = np.ones(n)
    
    Y = sparse.coo_matrix((data, (rows, cols)), shape=(n, k))
    
    return Y
