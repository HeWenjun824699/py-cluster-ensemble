import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

def eig1(A, c=None, isMax=1, isSym=1):
    """
    EIG1
    
    Args:
        A: matrix
        c: number of eigenvectors
        isMax: 1 for largest eigenvalues, 0 for smallest
        isSym: 1 for symmetric enforcement
    
    Returns:
        eigvec: eigenvectors corresponding to selected eigenvalues
        eigval: selected eigenvalues
        eigval_full: all eigenvalues (or sorted subset if partial)
    """
    
    if c is None:
        c = A.shape[0]
        
    if c > A.shape[0]:
        c = A.shape[0]

    # Enforce symmetry if requested
    if isSym == 1:
        # A = max(A, A')
        # Note: Matlab max(A, A') for matrices does element-wise max.
        # It is NOT (A+A')/2.
        # However, for symmetric matrices they should be similar. 
        # But 'max' ensures strict symmetry if one is slightly off? 
        # No, max(A, A') makes it symmetric ONLY if A_ij and A_ji are related.
        # If A is an affinity matrix, usually A_ij should equal A_ji.
        A = np.maximum(A, A.T)

    n = A.shape[0]
    
    # Try full decomposition first if dense or small, or strictly follow Matlab logic?
    # Matlab logic: try A=full(A); [v,d]=eig(A); catch ... eigs ...
    # Python numpy.linalg.eigh is very efficient for symmetric matrices.
    
    try:
        # We assume A is symmetric since isSym=1 default and we did np.maximum
        # Use eigh for symmetric matrices
        # subset_by_index can replace sorting manually if we know what we want.
        # isMax=1 -> Largest. isMax=0 -> Smallest.
        
        if isMax == 0:
            # Smallest c eigenvalues
            # indices: 0 to c-1
            vals, vecs = eigh(A, subset_by_index=[0, c-1])
            # They are returned in ascending order.
            eigval = vals
            eigvec = vecs
            
            # For 'eigval_full', the Matlab code returns ALL eigenvalues sorted.
            # This is expensive. If the caller doesn't need it, we might skip.
            # But to be consistent with 'eigval_full' return, we might need all?
            # The matlab function returns [eigvec, eigval, eigval_full].
            # If the caller only unpacks 2, we save time? 
            # Python functions return a tuple.
            # Let's check if we can get all efficiently.
            # If c < n, eigh with subset is faster.
            # If we need eigval_full, we must compute all.
            # In 'optimization.m', usage is: [F, ~, evs]=eig1(L, c, 0);
            # It DOES use the 3rd output 'evs' (eigval_full) in: evs(:,ii+1) = ev;
            # So we NEED all eigenvalues.
            
            vals_all, vecs_all = eigh(A) # Compute all
            
            # Sort is ascending by default for eigh
            # isMax=0 -> sort ascending
            idx = np.argsort(vals_all)
            d1 = vals_all[idx]
            v = vecs_all[:, idx]
            
            idx1 = slice(0, c)
            eigval = d1[idx1]
            eigvec = v[:, idx1]
            eigval_full = d1
            
        else:
            # Largest c
            vals_all, vecs_all = eigh(A)
            # Sort descending
            idx = np.argsort(vals_all)[::-1]
            d1 = vals_all[idx]
            v = vecs_all[:, idx]
            
            idx1 = slice(0, c)
            eigval = d1[idx1]
            eigvec = v[:, idx1]
            eigval_full = d1

    except Exception as e:
        # Fallback to sparse if memory error or otherwise
        # In Matlab: if isMax==0 -> 'sa', else 'la'
        if sp.issparse(A) is False:
             A = sp.csr_matrix(A)
             
        if isMax == 0:
            vals, vecs = eigsh(A, k=c, which='SA', tol=1e-5)
            # eigsh returns unsorted usually, or sorted?
            # Sorted in ascending usually for 'SA'
            idx = np.argsort(vals)
            eigval = vals[idx]
            eigvec = vecs[:, idx]
            eigval_full = eigval # Sparse cannot get all easily without k=n
        else:
            vals, vecs = eigsh(A, k=c, which='LA', tol=1e-5)
            idx = np.argsort(vals)[::-1]
            eigval = vals[idx]
            eigvec = vecs[:, idx]
            eigval_full = eigval

    return eigvec, eigval, eigval_full
