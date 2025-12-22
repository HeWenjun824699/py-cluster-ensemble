import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from .litekmeans import litekmeans

def Tcut_for_bipartite_graph(B, Nseg, maxKmIters=100, cntReps=3, random_state=None):
    """
    Tcut_for_bipartite_graph
    
    Parameters:
    -----------
    B : scipy.sparse matrix
        |X|-by-|Y| cross-affinity matrix (bipartite graph).
    Nseg : int
        Number of segments (clusters).
    maxKmIters : int, optional (default=100)
    cntReps : int, optional (default=3)
    random_state : int, RandomState instance or None, optional (default=None)
        
    Returns:
    --------
    labels : numpy.ndarray
        Cluster labels.
    """
    
    # B is |X|-by-|Y|
    Nx, Ny = B.shape
    
    if Ny < Nseg:
        raise ValueError('Need more columns!')
        
    # dx = sum(B, 2)
    # scipy sparse sum returns a matrix/array of shape (Nx, 1)
    dx = np.array(B.sum(axis=1)).flatten()
    dx[dx == 0] = 1e-10 # Avoid division by zero
    
    # Dx = sparse(1:Nx, 1:Nx, 1./dx)
    Dx = sparse.diags(1.0 / dx, 0, shape=(Nx, Nx))
    
    # Wy = B' * Dx * B
    Wy = B.T @ Dx @ B
    
    # Compute Ncut eigenvectors
    # normalized affinity matrix
    # d = sum(Wy, 2)
    d = np.array(Wy.sum(axis=1)).flatten()
    
    # D = sparse(1:Ny, 1:Ny, 1./sqrt(d))
    # Handle d=0 just in case, though usually connected
    d_sqrt = np.sqrt(d)
    d_sqrt[d_sqrt == 0] = 1e-10
    D = sparse.diags(1.0 / d_sqrt, 0, shape=(Ny, Ny))
    
    # nWy = D * Wy * D
    nWy = D @ Wy @ D
    
    # nWy = (nWy + nWy') / 2
    nWy = (nWy + nWy.T) / 2
    
    # Computer eigenvectors
    # Matlab: [evec, eval] = eig(full(nWy));
    # Use numpy eigh for symmetric matrices on dense version if small enough,
    # or scipy eigsh if sparse. Original uses 'full', implying dense is acceptable.
    # Assuming Ny (number of base clusters) is reasonable size (e.g. < few thousands).
    if sparse.issparse(nWy):
        nWy_dense = nWy.toarray()
    else:
        nWy_dense = nWy
        
    evals, evecs = np.linalg.eigh(nWy_dense)
    
    # Sort descending
    idx = np.argsort(evals)[::-1]
    # Ncut_evec = D * evec(:, idx(1:Nseg))
    # Python 0-based slicing: idx[:Nseg]
    Ncut_evec = D @ evecs[:, idx[:Nseg]]
    
    # Compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
    # evec = Dx * B * Ncut_evec
    evec = Dx @ B @ Ncut_evec
    
    # Normalize each row to unit norm
    # evec = bsxfun(@rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10)
    row_norms = np.sqrt(np.sum(evec**2, axis=1)) + 1e-10
    evec = evec / row_norms[:, np.newaxis]
    
    # k-means
    # labels = litekmeans(evec, Nseg)
    labels, _, _, _ = litekmeans(evec, Nseg, max_iter=maxKmIters, replicates=cntReps, random_state=random_state)
    
    return labels
