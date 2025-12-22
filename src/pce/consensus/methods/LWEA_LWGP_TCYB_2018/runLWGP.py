import numpy as np
from scipy.linalg import eigh
from scipy.sparse import spdiags, issparse

from .litekmeans import litekmeans


def runLWGP(bcs, baseClsSegs, ECI, clsNum):
    """
    Run Locally Weighted Graph Partitioning (LWGP).
    
    Args:
        bcs: N x M matrix of cluster labels (globally unique if coming from getAllSegs).
             (Note: Not directly used in the algorithm logic here, but passed in signature).
        baseClsSegs: Sparse matrix (nCls x N) indicating cluster membership.
        ECI: Vector of weights (nCls,).
        clsNum: Number of clusters (int) or list of ints.
        
    Returns:
        labels: Clustering results. N x len(clsNum).
    """
    # Ensure clsNum is iterable
    if np.isscalar(clsNum):
        ks = [clsNum]
    else:
        ks = clsNum
        
    N = baseClsSegs.shape[1]
    nCls = baseClsSegs.shape[0]
    
    # Build the locally weighted bipartite graph
    # Matlab: lwB = bsxfun(@times, baseClsSegs, ECI);
    # Scales each row i of baseClsSegs by ECI[i].
    # equivalent to: D * baseClsSegs where D is diag(ECI)
    
    # ECI is 1D array (nCls,)
    # Construct diagonal matrix
    D_ECI = spdiags(ECI, 0, nCls, nCls)
    
    # lwB is nCls x N
    lwB = D_ECI.dot(baseClsSegs)
    
    # B passed to partitioning is lwB' -> N x nCls
    B = lwB.transpose()
    
    results = np.zeros((N, len(ks)), dtype=int)
    
    for i, K in enumerate(ks):
        labels = bipartiteGraphPartitioning(B, K)
        results[:, i] = labels
        
    return results

def bipartiteGraphPartitioning(B, Nseg):
    """
    Partition the bipartite graph.
    
    Args:
        B: N x nCls sparse matrix (cross-affinity).
        Nseg: Number of clusters.
        
    Returns:
        labels: Cluster labels (N,).
    """
    Nx, Ny = B.shape
    
    if Ny < Nseg:
        raise ValueError('The cluster number is too large!')
        
    # dx = sum(B, 2)
    dx = np.array(B.sum(axis=1)).flatten()
    dx[dx == 0] = 1e-10
    
    # Dx = sparse(1:Nx, 1:Nx, 1./dx)
    Dx = spdiags(1.0 / dx, 0, Nx, Nx)
    
    # Wy = B' * Dx * B  -> (Ny x Nx) * (Nx x Nx) * (Nx x Ny) -> Ny x Ny
    Wy = B.transpose().dot(Dx).dot(B)
    
    # Normalized affinity matrix
    # d = sum(Wy, 2)
    d = np.array(Wy.sum(axis=1)).flatten()
    
    # D = sparse(1:Ny, 1:Ny, 1./sqrt(d))
    # Handle division by zero if any row sum is 0 (unlikely in connected components, but safe to check)
    d_sqrt_inv = 1.0 / np.sqrt(d)
    d_sqrt_inv[np.isinf(d_sqrt_inv)] = 0
    D = spdiags(d_sqrt_inv, 0, Ny, Ny)
    
    # nWy = D * Wy * D
    nWy = D.dot(Wy).dot(D)
    
    # nWy = (nWy + nWy') / 2
    # Ensure symmetry
    nWy = (nWy + nWy.transpose()) / 2
    
    # Compute eigenvectors
    # Matlab: [evec, eval] = eig(full(nWy))
    # sort desc
    # In Python eigh returns eigenvalues in ascending order.
    
    if issparse(nWy):
        nWy_dense = nWy.toarray()
    else:
        nWy_dense = nWy
        
    # eigh for Hermitian/Symmetric matrices
    vals, vecs = eigh(nWy_dense)
    
    # Sort descending
    idx = np.argsort(vals)[::-1]
    # vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Ncut_evec = D * evec(:, 1:Nseg)
    # Top Nseg eigenvectors
    Ncut_evec = D.dot(vecs[:, :Nseg])
    
    # Compute Ncut eigenvectors on the entire bipartite graph (transfer!)
    # evec = Dx * B * Ncut_evec
    evec = Dx.dot(B).dot(Ncut_evec)
    
    # Normalize each row to unit norm
    # Matlab: bsxfun(@rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10)
    # sklearn.preprocessing.normalize does this (L2 norm)
    # But let's follow the formula strictly to match epsilon handling if needed.
    # row_norms = sqrt(sum(x^2))
    row_norms = np.sqrt(np.sum(evec**2, axis=1)) + 1e-10
    evec_norm = evec / row_norms[:, np.newaxis]
    
    # k-means
    # labels = litekmeans(evec, Nseg)
    labels, _, _, _, _ = litekmeans(evec_norm, Nseg)
    
    return labels + 1 # Return 1-based labels
