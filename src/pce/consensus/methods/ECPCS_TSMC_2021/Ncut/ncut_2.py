import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def ncut_2(W, nbEigenValues=8, dataNcut=None):
    """
    Compute Continuous Ncut eigenvectors.
    
    Corresponds to ncut_2.m
    
    Parameters:
    -----------
    W : numpy.ndarray or scipy.sparse.csr_matrix
        Symmetric similarity matrix.
    nbEigenValues : int
        Number of eigenvectors to compute.
    dataNcut : dict, optional
        Parameters dictionary.
        
    Returns:
    --------
    Eigenvectors : numpy.ndarray
        (n, nbEigenValues) matrix.
    Eigenvalues : numpy.ndarray
        (nbEigenValues,) array.
    """
    
    if dataNcut is None:
        dataNcut = {
            'offset': 5e-1,
            'verbose': 0,
            'maxiterations': 300,
            'eigsErrorTolerance': 1e-8,
            'valeurMin': 1e-6
        }
        
    # Set defaults if keys missing
    dataNcut.setdefault('offset', 5e-1)
    dataNcut.setdefault('verbose', 0)
    dataNcut.setdefault('maxiterations', 300)
    dataNcut.setdefault('eigsErrorTolerance', 1e-8)
    dataNcut.setdefault('valeurMin', 1e-6)
    
    # make W matrix sparse and sparsify
    # W = sparsifyc(W,dataNcut.valeurMin);
    # Keeps values > valeurMin
    if not sparse.issparse(W):
        W = sparse.csr_matrix(W)
        
    # Efficient sparsify
    W.data[W.data <= dataNcut['valeurMin']] = 0
    W.eliminate_zeros()
    
    # check for matrix symmetry
    # if max(max(abs(W-W'))) > 1e-10
    # Checking symmetry on sparse matrix can be expensive. 
    # Assume symmetric or quick check if small.
    # For large W, trust user or check diff.
    # Let's skip expensive check for performance, or do a quick check on sample?
    # Original code errors out. We'll assume valid input or basic check.
    if np.abs(W - W.T).max() > 1e-10:
        raise ValueError('W not symmetric')
        
    n = W.shape[0]
    nbEigenValues = min(nbEigenValues, n)
    offset = dataNcut['offset']
    
    # degrees and regularization
    # d = sum(abs(W),2);
    # W is non-negative usually, but abs just in case
    d = np.array(np.abs(W).sum(axis=1)).flatten()
    
    # dr = 0.5 * (d - sum(W,2));
    dw = np.array(W.sum(axis=1)).flatten()
    dr = 0.5 * (d - dw)
    
    # d = d + offset * 2;
    d = d + offset * 2
    
    # dr = dr + offset;
    dr = dr + offset
    
    # W = W + spdiags(dr,0,n,n);
    W = W + sparse.spdiags(dr, 0, n, n)
    
    # Dinvsqrt = 1./sqrt(d+eps);
    Dinvsqrt = 1.0 / np.sqrt(d + np.finfo(float).eps)
    
    # P = spmtimesd(W,Dinvsqrt,Dinvsqrt);
    # Computes Dinvsqrt * W * Dinvsqrt
    # Dinvsqrt is diagonal.
    # Element-wise: P_ij = W_ij * D_i * D_j
    
    # Efficient way using sparse matrix multiplication
    # D_mat = sparse.spdiags(Dinvsqrt, 0, n, n)
    # P = D_mat @ W @ D_mat
    
    # Or more efficient: scale rows then cols
    # P = W.copy()
    # Scipy sparse diags multiplication
    D_mat = sparse.spdiags(Dinvsqrt, 0, n, n)
    P = D_mat.dot(W).dot(D_mat)
    
    # [vbar,s,convergence] = eigs(@mex..., size(P,1), nbEigenValues, 'LA', ...);
    # 'LA': Largest Algebraic
    
    # eigsh for symmetric matrices
    # returns eigenvalues, eigenvectors
    # Default uses 'LM' (Largest Magnitude). 'LA' is equivalent to 'LA' in eigs.
    # Note: eigsh returns eigenvalues in ascending order.
    
    # We want largest algebraic.
    
    try:
        eigvals, eigvecs = eigsh(P, k=nbEigenValues, which='LA', 
                                 tol=dataNcut['eigsErrorTolerance'], 
                                 maxiter=dataNcut['maxiterations'])
    except sparse.linalg.ArpackNoConvergence as e:
        # If not converged, use what we have
        eigvals = e.eigenvalues
        eigvecs = e.eigenvectors
        
    # Matlab: [x,y] = sort(-s); Eigenvalues = -x; (Sort descending)
    # eigsh returns ascending.
    idx = np.argsort(eigvals)[::-1]
    Eigenvalues = eigvals[idx]
    vbar = eigvecs[:, idx]
    
    # Eigenvectors = spdiags(Dinvsqrt,0,n,n) * vbar;
    Eigenvectors = D_mat.dot(vbar)
    
    # Normalize
    # for i=1:size(Eigenvectors,2)
    #    Eigenvectors(:,i) = (Eigenvectors(:,i) / norm(Eigenvectors(:,i))  )*norm(ones(n,1));
    #    if Eigenvectors(1,i)~=0
    #        Eigenvectors(:,i) = - Eigenvectors(:,i) * sign(Eigenvectors(1,i));
    #    end
    # end
    
    norm_ones = np.linalg.norm(np.ones(n))
    
    for i in range(Eigenvectors.shape[1]):
        vec = Eigenvectors[:, i]
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec = (vec / nrm) * norm_ones
            
        if vec[0] != 0:
            vec = -vec * np.sign(vec[0])
            
        Eigenvectors[:, i] = vec
        
    return Eigenvectors, Eigenvalues
