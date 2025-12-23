import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

def eig1(A, c=None, isMax=1, isSym=1):
    """
    EIG1 Equivalent to MATLAB eig1 function.
    Optimized for symmetric matrices using eigh and subset computation.
    
    Parameters:
    A: numpy array or matrix
    c: number of eigenvalues/vectors to return
    isMax: 1 to sort descending (largest), 0 to sort ascending (smallest)
    isSym: 1 to enforce symmetry
    
    Returns:
    eigvec: selected eigenvectors (c columns)
    eigval: selected eigenvalues (c values)
    eigval_full: available eigenvalues (may be partial if optimized)
    """
    A = np.array(A)
    n = A.shape[0]
    
    if c is None:
        c = n
    elif c > n:
        c = n
        
    # Optimization: If symmetric and we need a subset
    if isSym == 1:
        # Check if we can use partial decomposition
        # Optimize.py needs up to c+1 eigenvalues (indices 0 to c) for checking convergence
        # So we need at least c+1 values.
        
        k_needed = c + 1
        if k_needed > n:
            k_needed = n
            
        # Use eigh with subset_by_index
        # subset_by_index=(start, end) inclusive
        
        if isMax == 0:
            # Ascending, we want smallest
            # Get indices 0 to k_needed-1
            subset = (0, k_needed - 1)
            
            try:
                # eigh is much faster for symmetric matrices
                d, v = scipy.linalg.eigh(A, subset_by_index=subset)
            except ValueError:
                # Fallback if subset_by_index is not supported (unlikely in recent scipy)
                # or other error
                d, v = scipy.linalg.eigh(A)
        else:
            # Descending, we want largest
            # eigh returns ascending. Largest are at the end.
            # Indices n-k_needed to n-1
            subset = (n - k_needed, n - 1)
            try:
                d, v = scipy.linalg.eigh(A, subset_by_index=subset)
            except ValueError:
                d, v = scipy.linalg.eigh(A)
            
            # Reverse to make descending
            d = d[::-1]
            v = v[:, ::-1]

        # Select the requested c
        # d contains k_needed values
        eigval = d[:c]
        eigvec = v[:, :c]
        eigval_full = d # This contains c+1 values if optimization worked
        
        return eigvec, eigval, eigval_full

    # Fallback for non-symmetric or if something failed (though we handled sym above)
    # Original logic for non-symmetric A
    
    if isSym == 1:
        # Should have been handled above, but just in case
        A = np.maximum(A, A.T)
        
    try:
        # Dense eig
        d, v = scipy.linalg.eig(A)
    except Exception:
        # Sparse eig
        A_sparse = scipy.sparse.csc_matrix(A)
        if isMax == 0:
            d, v = scipy.sparse.linalg.eigs(A_sparse, k=c, which='SR', tol=1e-5)
        else:
            d, v = scipy.sparse.linalg.eigs(A_sparse, k=c, which='LR', tol=1e-5)
            
    if np.all(np.isreal(d)):
        d = d.real
    
    if isMax == 0:
        idx = np.argsort(d) # Ascending
    else:
        idx = np.argsort(d)[::-1] # Descending

    d1 = d[idx]
    
    # Return requested c
    eigval = d1[:c]
    eigvec = v[:, idx[:c]]
    
    eigval_full = d1 # All values

    return eigvec, eigval, eigval_full
