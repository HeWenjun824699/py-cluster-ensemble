import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import linalg as la

def eig1(A, c=None, isMax=1, isSym=1):
    """
    Eigen-decomposition of matrix A.
    
    Parameters:
    A : numpy.ndarray or scipy.sparse matrix
    c : int, optional
        Number of eigenvalues/vectors to return. Default is size of A.
    isMax : int, optional
        If 1, return largest eigenvalues (default). If 0, return smallest.
    isSym : int, optional
        If 1, enforce symmetry (default).
        
    Returns:
    eigvec : numpy.ndarray
        Eigenvectors.
    eigval : numpy.ndarray
        Eigenvalues.
    eigval_full : numpy.ndarray
        All eigenvalues (or top c if truncated).
    """
    if c is None:
        c = A.shape[0]
    elif c > A.shape[0]:
        c = A.shape[0]
        
    if isSym == 1:
        # A = max(A, A')
        # In MATLAB, max(A, A') for matrices compares element-wise.
        # Ensure A is dense or sparse compatible
        if sparse.issparse(A):
            A = A.maximum(A.T)
        else:
            A = np.maximum(A, A.T)
            
    # Try dense decomposition first if A is not too large or if it's dense
    try:
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
            
        # MATLAB: [v, d] = eig(A); d = diag(d);
        # Python scipy.linalg.eigh for symmetric, eig for non-symmetric
        if isSym:
            d, v = la.eigh(A_dense)
        else:
            d, v = la.eig(A_dense)
            d = np.real(d) # Keep real part like MATLAB usually implies for symmetric/Laplacian
            
    except MemoryError:
        # Fallback to sparse solver
        # MATLAB: eigs(sparse(A), c, 'sa'/'la')
        if not sparse.issparse(A):
            A = sparse.csr_matrix(A)
            
        if isMax == 0:
            # Smallest Algebraic
            # Note: 'SA' in scipy matches 'sa' in MATLAB
            d, v = spla.eigsh(A, k=c, which='SA', tol=1e-5)
        else:
            # Largest Algebraic
            d, v = spla.eigsh(A, k=c, which='LA', tol=1e-5)
            
    # Sort eigenvalues and vectors
    # MATLAB d is usually not sorted by default in eig, but sorted in eigs
    # We sort manually to be safe matches MATLAB 'sort' behavior
    
    if isMax == 0:
        idx = np.argsort(d) # Ascending
    else:
        idx = np.argsort(d)[::-1] # Descending
        
    d1 = d[idx]
    v1 = v[:, idx]
    
    # Select top c
    if c > len(d1):
        c = len(d1)
    
    idx1 = range(c)
    eigval = d1[idx1]
    eigvec = v1[:, idx1]
    
    eigval_full = d1
    
    return eigvec, eigval, eigval_full
