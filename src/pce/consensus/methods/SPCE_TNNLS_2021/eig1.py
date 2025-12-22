import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

def eig1(A, c=None, isMax=1, isSym=1):
    """
    EIG1 Equivalent to MATLAB eig1 function.
    
    Parameters:
    A: numpy array or matrix
    c: number of eigenvalues/vectors to return (default: all)
    isMax: 1 to sort descending (largest), 0 to sort ascending (smallest)
    isSym: 1 to enforce symmetry
    
    Returns:
    eigvec: selected eigenvectors
    eigval: selected eigenvalues
    eigval_full: all sorted eigenvalues
    """
    A = np.array(A)
    n = A.shape[0]
    
    if c is None:
        c = n
    elif c > n:
        c = n
        
    if isSym == 1:
        A = np.maximum(A, A.T)
        
    try:
        # Dense eig
        # MATLAB: A=full(A); [v, d] = eig(A); d = diag(d);
        # Python returns w (vals), v (vecs)
        d, v = scipy.linalg.eig(A)
        
    except Exception:
        # Sparse eig
        A_sparse = scipy.sparse.csc_matrix(A)
        # MATLAB eigs options: 'sa' (Smallest Algebraic), 'la' (Largest Algebraic)
        # Scipy: 'SR' (Smallest Real), 'LR' (Largest Real) usually used for symmetric.
        # But eigs is for general. eigsh is for symmetric.
        # Following MATLAB 'eigs' generic call structure:
        
        if isMax == 0:
            d, v = scipy.sparse.linalg.eigs(A_sparse, k=c, which='SR', tol=1e-5)
        else:
            d, v = scipy.sparse.linalg.eigs(A_sparse, k=c, which='LR', tol=1e-5)
            
    # Sorting
    # MATLAB: if isMax == 0: sort(d) (ascending)
    # else: sort(d, 'descend')
    
    # Handle complex return types from eig/eigs
    if np.all(np.isreal(d)):
        d = d.real
    
    if isMax == 0:
        idx = np.argsort(d) # Ascending
    else:
        idx = np.argsort(d)[::-1] # Descending (argsort gives ascending, reverse it)

    d1 = d[idx]
    
    idx1 = idx[:c]
    
    eigval = d[idx1]
    eigvec = v[:, idx1]
    
    eigval_full = d[idx]

    return eigvec, eigval, eigval_full
