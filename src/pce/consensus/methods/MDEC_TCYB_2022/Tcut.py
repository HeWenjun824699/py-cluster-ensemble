import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def Tcut(B, Nseg):
    """
    Corresponds to Tcut.m
    
    Args:
        B: (Nx, Ny) matrix. In performBG, B is weightedB', so Nx=N, Ny=nCls.
        Nseg: Number of segments (clusters).
    """
    Nx, Ny = B.shape
    if Ny < Nseg:
        raise ValueError('Need more superpixels!')
        
    # dx = sum(B,2);
    dx = np.sum(B, axis=1)
    # dx(dx==0) = eps; 
    # Ensure dx is flat array for operations
    dx = np.array(dx).flatten()
    dx[dx == 0] = np.finfo(float).eps
    
    # Dx = sparse(1:Nx,1:Nx,1./dx); 
    Dx = sp.diags(1.0 / dx)
    
    # Wy = B'*Dx*B;
    # B might be dense or sparse. If dense, B.T is dense.
    # Dx is sparse.
    # If B is numpy array:
    Wy = B.T @ Dx @ B
    
    # d = sum(Wy,2);
    d = np.sum(Wy, axis=1)
    # d(d==0) = eps;
    d = np.array(d).flatten()
    d[d == 0] = np.finfo(float).eps
    
    # D = sparse(1:Ny,1:Ny,1./sqrt(d)); 
    D = sp.diags(1.0 / np.sqrt(d))
    
    # nWy = D*Wy*D; 
    nWy = D @ Wy @ D
    
    # nWy = (nWy+nWy')/2;
    if sp.issparse(nWy):
        nWy = (nWy + nWy.T) / 2
    else:
        nWy = (nWy + nWy.T) / 2
        
    # [evec,eval] = eig(full(nWy));
    # Use eigh for symmetric matrices. Returns eigenvalues in ascending order.
    if sp.issparse(nWy):
        nWy_dense = nWy.toarray()
    else:
        nWy_dense = nWy
        
    evals, evecs = eigh(nWy_dense)
    
    # [~,idx] = sort(diag(eval),'descend');
    idx = np.argsort(evals)[::-1]
    
    # Ncut_evec = D*evec(:,idx(1:Nseg));
    # 1:Nseg means top Nseg eigenvectors
    top_indices = idx[:Nseg]
    Ncut_evec = D @ evecs[:, top_indices]
    
    # evec = Dx * B * Ncut_evec;
    evec = Dx @ B @ Ncut_evec
    
    # normalize each row to unit norm
    # evec = bsxfun( @rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10 );
    row_norms = np.sqrt(np.sum(evec**2, axis=1)) + 1e-10
    evec = evec / row_norms[:, np.newaxis]
    
    # labels = kmeans(evec,Nseg,'MaxIter',100,'Replicates',3);
    # Replicates=3 -> n_init=3
    kmeans = KMeans(n_clusters=Nseg, max_iter=100, n_init=3)
    labels = kmeans.fit_predict(evec)
    
    # Matlab labels are usually 1-based. Python 0-based.
    # We return 1-based labels to be strictly consistent with Matlab output expectation if needed,
    # or just the labels. Usually for clustering result comparison it doesn't matter (mapped),
    # but let's return 1-based to mimic Matlab behavior.
    return labels + 1
