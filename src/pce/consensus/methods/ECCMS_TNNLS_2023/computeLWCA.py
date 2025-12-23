import numpy as np
from scipy.sparse import diags

def computeLWCA(baseClsSegs, ECI, M):
    """
    Get locally weighted co-association matrix.
    
    Args:
        baseClsSegs: Sparse matrix (nCls x N).
        ECI: ECI vector (nCls).
        M: Number of base clusterings.
        
    Returns:
        LWCA: Locally Weighted Co-Association Matrix (N x N).
    """
    # Matlab: LWCA = (bsxfun(@times, baseClsSegs, ECI')) * baseClsSegs' / M;
    # baseClsSegs (N x nCls) in Matlab (transposed).
    # Here baseClsSegs is nCls x N.
    # We want (ECI weighted baseClsSegs.T) @ baseClsSegs / M ?
    # Wait, Matlab: baseClsSegs (N x nCls) .* ECI' (1 x nCls).
    # This weights the columns of the N x nCls matrix.
    # Which corresponds to rows of our nCls x N matrix.
    
    # So we weight the rows of baseClsSegs by ECI.
    
    D = diags(ECI) # nCls x nCls diagonal matrix
    WeightedBaseClsSegs = D @ baseClsSegs # nCls x N
    
    # LWCA = WeightedBaseClsSegs.T @ baseClsSegs / M ??
    # Matlab: (Weighted H') * H' ?? No.
    # Matlab: LWCA = (Weighted H') * H / M; (where H is original nCls x N, H' is N x nCls)
    # Actually Matlab:
    # baseClsSegs = baseClsSegs'; % Becomes N x nCls
    # LWCA = (Weighted N x nCls) * (N x nCls)'; 
    # LWCA = (Weighted N x nCls) * (nCls x N); -> N x N.
    
    # So in Python with nCls x N:
    # Weighted rows -> (WeightedBaseClsSegs).T @ baseClsSegs / M?
    # No.
    # Matrix mult: (N x nCls) @ (nCls x N).
    # The first one is weighted.
    # So: (baseClsSegs.T scaled by ECI) @ baseClsSegs / M.
    # Scaling baseClsSegs.T by ECI (1 x nCls) means scaling columns.
    # Which means scaling rows of baseClsSegs.
    
    # So yes: (D @ baseClsSegs).T @ baseClsSegs / M?
    # Let's verify:
    # (D @ H).T = H.T @ D.T = H.T @ D.
    # So H.T @ D @ H / M.
    # This matches the logic "Weighted Co-Association".
    
    WeightedH = D @ baseClsSegs
    LWCA = baseClsSegs.T @ WeightedH / M
    
    # Convert to dense
    LWCA = LWCA.toarray()
    
    # LWCA = LWCA - diag(diag(LWCA)) + eye(N);
    np.fill_diagonal(LWCA, 1.0)
    
    return LWCA
