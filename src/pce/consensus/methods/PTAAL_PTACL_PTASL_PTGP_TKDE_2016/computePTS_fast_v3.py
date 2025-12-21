import numpy as np
from scipy import sparse

def compute_pts_fast_v3(S, mc_labels, para):
    """
    Compute the probability trajectory based similarity matrix.
    
    Args:
        S: N x N Co-association matrix.
        mc_labels: N x 2 array mapping objects to microclusters.
        para: Dictionary with keys 'K' and 'T'.
        
    Returns:
        Sim: Similarity matrix.
    """
    K = para['K']
    T = para['T']
    
    N = S.shape[0]
    
    # mcSizes calculation
    # Count how many original objects belong to each microcluster.
    # mc_labels[:, 1] contains 1-based microcluster IDs.
    mc_counts = np.bincount(mc_labels[:, 1])[1:] # skip 0 index
    
    # Ensure mc_counts matches N
    if len(mc_counts) < N:
        tmp = np.zeros(N)
        tmp[:len(mc_counts)] = mc_counts
        mc_counts = tmp
    elif len(mc_counts) > N:
        mc_counts = mc_counts[:N]

    mc_sizes = mc_counts.reshape(-1, 1) # Column vector

    # Thresholding
    thres_pos = int(N - np.floor(K) + 1)
    if thres_pos > N: thres_pos = N
    if thres_pos < 1: thres_pos = 1
    
    # Zero diagonal explicitly before thresholding logic
    np.fill_diagonal(S, 0)
    
    # Sort to find thresholds (ascending)
    sorted_s = np.sort(S, axis=1)
    
    # thresholds = sortedS[:, thresPos] (Matlab 1-based) -> Python thres_pos - 1
    thresholds = sorted_s[:, thres_pos - 1]
    
    # Apply threshold: Keep elements >= threshold
    mask = S < thresholds[:, np.newaxis]
    S[mask] = 0
    
    # Symmetrize
    S = np.maximum(S, S.T)
    
    # Weighting: S = S * mcSizes' (Broadcast)
    S = S * mc_sizes.T
    
    # Row normalize
    row_sum = np.sum(S, axis=1)
    isolated_idx = np.where(row_sum == 0)[0]
    
    # Avoid division by zero
    row_sum[isolated_idx] = -1 
    
    # P = S / rowSum
    P = S / row_sum[:, np.newaxis]
    
    # Facilitate Computation of PTS (Dense Optimized)
    # Using dense matrices to avoid slowdown from fill-in in sparse matrices
    P_dense = P
    tmp_P = P_dense.copy()
    
    # inProdP = tmpP * P'
    in_prod_P = tmp_P.dot(P_dense.T)
    
    # Loop T-1 times
    for ii in range(1, int(T)):
        tmp_P = tmp_P.dot(P_dense)
        in_prod_P = in_prod_P + tmp_P.dot(tmp_P.T)
        
    # Normalize
    diag_val = np.diag(in_prod_P)
    in_prod_Pii = np.tile(diag_val[:, np.newaxis], (1, N))
    
    # Sim = inProdP ./ sqrt(inProdPii .* inProdPii')
    # in_prod_Pii.T[i,j] = diag_val[j]
    denom = np.sqrt(in_prod_Pii * in_prod_Pii.T)
    
    # Avoid div by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        Sim = in_prod_P / denom
        
    Sim[np.isnan(Sim)] = 0
    
    # Handle isolated points
    Sim[isolated_idx, :] = 1e-20
    Sim[:, isolated_idx] = 1e-20
    
    return Sim