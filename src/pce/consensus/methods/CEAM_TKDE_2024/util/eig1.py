import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy import linalg

def eig1(A, c=None, isMax=1, isSym=1):
    """
    Eigen-decomposition wrapper.
    
    A: Matrix
    c: Number of eigenvalues/vectors
    isMax: 1 for largest, 0 for smallest
    isSym: 1 for symmetric
    """
    if c is None:
        c = A.shape[0]
    
    if c > A.shape[0]:
        c = A.shape[0]
        
    if isSym:
        # A = max(A, A')
        # Note: In MATLAB max(A, A') operates element-wise.
        # Check if A is sparse
        if sparse.issparse(A):
            # Element-wise max for sparse matrices is not directly 'max' in old scipy versions,
            # but A.maximum(A.T) works.
            A = A.maximum(A.T)
        else:
            A = np.maximum(A, A.T)
            
    try:
        # Try full dense eig
        # In MATLAB: [v, d] = eig(A); d = diag(d);
        if sparse.issparse(A):
            full_A = A.toarray()
        else:
            full_A = A
            
        d, v = linalg.eig(full_A)
        # Check if complex (should be real if symmetric, but precision issues)
        if isSym:
            d = np.real(d)
            v = np.real(v)
            
    except Exception:
        # Sparse eig
        # if isMax == 0: 'sa'
        # else: 'la'
        if isMax == 0:
            d, v = splinalg.eigs(A, k=c, which='SR', tol=1e-5) # 'SR' (Smallest Real) is often better for symmetric/Laplacian
            # Re-check 'sa' in MATLAB (Smallest Algebraic). 'SR' is Smallest Real part. 'SA' is Smallest Algebraic.
            # Using 'SA' to match MATLAB 'sa'.
            # Note: scipy eigs 'SA' might assume symmetric? Use eigsh if symmetric.
            # But the code uses `eigs` even if `isSym=1` in the try-catch block for large matrices? 
            # Actually, the MATLAB code uses `eigs(sparse(A), c, 'sa', ...)`
            
            # Since we enforced symmetry above (if isSym), `eigsh` is preferred for stability,
            # but strictly following the code which calls `eigs` even for symmetric A (inside catch block).
            # However, `eigs` in scipy handles general matrices.
            pass
        
        # To strictly follow logic:
        # MATLAB code tries `eig` (dense) first. If it fails (memory?), it falls back to `eigs`.
        # Python `linalg.eig` usually handles whatever fits in memory.
        # I'll stick to the logic: Dense first, then Sparse if exception (unlikely in Python unless OOM, which kills process usually, but maybe for sparse format errors?)
        
        # Let's assume we want to use sparse solver if explicitly sparse and large, 
        # but the MATLAB code `try A=full(A)... catch ...` implies it prefers dense unless it fails.
        
        # If we are here, dense failed.
        if isMax == 0:
             d, v = splinalg.eigs(A, k=c, which='SR', tol=1e-5) # Approximation of 'sa'
        else:
             d, v = splinalg.eigs(A, k=c, which='LR', tol=1e-5) # Approximation of 'la'
             
        if isSym:
            d = np.real(d)
            v = np.real(v)

    # Sort
    if isMax == 0:
        idx = np.argsort(d)
    else:
        idx = np.argsort(d)[::-1] # Descending
        
    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:, idx1]
    
    eigval_full = d[idx]
    
    return eigvec, eigval, eigval_full
