import numpy as np
from scipy.sparse import eye, diags

def computeLWCA(baseClsSegs, ECI, M):
    """
    Get locally weighted co-association matrix.
    
    Args:
        baseClsSegs: Sparse matrix (nCls x N).
        ECI: Vector of weights (nCls).
        M: Number of base clusterings.
        
    Returns:
        LWCA: N x N Co-association matrix.
    """
    # Matlab: baseClsSegs = baseClsSegs' -> N x nCls
    # baseClsSegs input is nCls x N
    # Let's transpose it
    baseClsSegs_T = baseClsSegs.transpose() # N x nCls
    
    N = baseClsSegs_T.shape[0]
    
    # Matlab: LWCA = (bsxfun(@times, baseClsSegs, ECI')) * baseClsSegs' / M;
    # baseClsSegs is N x nCls. ECI' is 1 x nCls.
    # Multiply each column j of baseClsSegs by ECI[j].
    
    # In sparse matrix multiplication, we can use diagonal matrix for broadcasting
    # A * D where D is diag(ECI).
    # Or just element-wise multiply if dense, but this is sparse.
    # baseClsSegs_T.multiply(ECI) works if ECI is broadcastable?
    # ECI is 1D array. csr_matrix.multiply broadcasts over rows?
    # Actually: A (N x K) * diag(v) (K x K) scales columns of A.
    
    # Construct diagonal matrix from ECI
    D = diags(ECI)
    
    # Weighted matrix (N x nCls)
    WeightedSegs = baseClsSegs_T.dot(D)
    
    # LWCA = WeightedSegs * baseClsSegs_T' / M
    #      = WeightedSegs * baseClsSegs / M
    LWCA = WeightedSegs.dot(baseClsSegs) / M
    
    # LWCA is likely dense now (N x N), or sparse. 
    # If N is large, this might be huge. Matlab code returns full matrix implicitly?
    # Matlab's LWCA = ... usually produces dense if not careful, but sparse * sparse is sparse.
    # However, the diagonal op suggests it's treated as a matrix.
    # Let's convert to dense if N is reasonable, or keep sparse.
    # Matlab code: LWCA = LWCA - diag(diag(LWCA)) + eye(N)
    
    # If sparse, getting diag is easy.
    # We should return a dense matrix if strictly mimicking Matlab's likely output for typical clustering tasks,
    # but sparse is safer for memory.
    # However, standard clustering (spectral, linkage) often needs full matrix or specific format.
    # Let's keep it defined by the operations.
    
    # Convert to dense for diagonal operations if it fits in memory?
    # Let's try to do it sparsely or densify.
    LWCA = LWCA.toarray()
    
    # LWCA = LWCA - diag(diag(LWCA)) + eye(N)
    np.fill_diagonal(LWCA, 1.0)
    
    return LWCA
