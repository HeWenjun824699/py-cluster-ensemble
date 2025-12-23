import numpy as np

def getLWCA(baseClsSegs, ECI, M):
    """
    Corresponds to getLWCA.m
    
    Args:
        baseClsSegs: (nCls, N)
        ECI: (nCls,) or (nCls, 1)
        M: Number of base clusterings
    """
    # baseClsSegs = baseClsSegs';
    S = baseClsSegs.T # (N, nCls)
    N = S.shape[0]
    
    # LWCA = (baseClsSegs.*repmat(ECI',N,1)) * baseClsSegs' / M;
    # ECI is likely (nCls,) or (nCls, 1). ECI' would be (1, nCls).
    # repmat(ECI', N, 1) repeats the row vector N times -> (N, nCls).
    # So we multiply S (N, nCls) element-wise with ECI broadcasted to rows.
    
    # Flatten ECI to 1D array of shape (nCls,) to match columns of S
    eci_flat = np.array(ECI).flatten()
    
    # In numpy: (N, nCls) * (nCls,) broadcasts the 1D array to every row
    weighted_S = S * eci_flat
    
    # * baseClsSegs' (which is S')
    # weighted_S (N, nCls) @ S.T (nCls, N) -> (N, N)
    LWCA = weighted_S @ S.T / M
    
    # LWCA = LWCA-diag(diag(LWCA))+eye(N);
    # Replace diagonal elements with 1
    np.fill_diagonal(LWCA, 1.0)
    
    return LWCA
