import numpy as np
from .norml import norml


def clbtocl(clb, p=None):
    """
    Cluster Binary to Cluster Labels.
    Assigns objects to clusters based on maximum posterior probability.
    
    MATLAB equivalent:
    function [cl, index] = clbtocl(clb,p)
    """
    randbreakties = 1
    
    # clb is (n_clusters, n_samples)
    
    # allzerocolumns = find(sum(clb,1)==0);
    # sum(clb, 1) in MATLAB is sum of columns (resulting in row vector).
    # wait. MATLAB sum(A, 1) sums distinct rows for each column. Result is 1 x Cols.
    # Yes.
    
    col_sums = np.sum(clb, axis=0)
    allzerocolumns = np.where(col_sums == 0)[0]
    
    if len(allzerocolumns) > 0:
        print(f"clbtocl: {len(allzerocolumns)} objects ({100*len(allzerocolumns)/clb.shape[1]:.0f}%) with all zero associations")
        clb[:, allzerocolumns] = np.random.rand(clb.shape[0], len(allzerocolumns))
        
    if randbreakties:
        clb = clb + np.random.rand(*clb.shape) / 10000.0
        
    # clb = norml(clb',1)';
    # Normalizing columns so they sum to 1 (conceptually).
    # MATLAB: norml(clb', 1) normalizes rows of clb'.
    clb_t = clb.T
    clb_norm_t = norml(clb_t, 1)
    clb = clb_norm_t.T
    
    # m = max(clb,[],1);
    # Max probability for each sample across all clusters
    m = np.max(clb, axis=0)
    
    n_samples = clb.shape[1]
    n_clusters = clb.shape[0]
    
    cl = np.zeros(n_samples, dtype=int)
    winnersprop = np.zeros(n_samples)
    
    # MATLAB loop: for i=size(clb,1):-1:1
    # Note: MATLAB is 1-based. If we want 0-based labels, we map row index directly.
    # The logic is just "who is the winner".
    # Iterate backwards to match MATLAB tie-breaking preference? 
    # MATLAB: if ties, find returns indices. overwriting happens?
    # `cl(a) = i`. If `a` was set by `i+1` before, it gets overwritten by `i`.
    # So lower index wins if traversed backwards?
    # Actually, `find(m==clb(i,:))` checks if the max value equals the value at `i`.
    # If multiple rows have the max value, `a` will include that sample for both `i`s.
    # If we iterate backwards (N down to 1), `i=N` sets `cl`. `i=N-1` sets `cl`.
    # If a sample has max at both N and N-1, `cl` is overwritten by N-1.
    # So lowest index wins.
    
    for i in range(n_clusters - 1, -1, -1):
        # find samples where cluster i is the winner
        # Use simple tolerance for float comparison or exact match since we used m = max(...)
        a = np.where(m == clb[i, :])[0]
        
        if len(a) > 0:
            cl[a] = i 
            winnersprop[a] = clb[i, a]
        
    if p is None:
        index = []
    else:
        index = np.where(winnersprop < p)[0]
        
    # print(f"clbtocl: delivering {np.max(cl) + 1} clusters")
    # print(f"clbtocl: average posterior prob is {np.mean(winnersprop)}")
    
    return cl, index
