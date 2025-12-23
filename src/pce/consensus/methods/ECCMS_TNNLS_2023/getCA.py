import numpy as np

def getCA(baseClsSegs, M):
    """
    Get Co-Association matrix.
    
    Args:
        baseClsSegs: Sparse matrix (nCls x N).
        M: Number of base clusterings.
        
    Returns:
        CA: Co-Association Matrix (N x N).
    """
    # CA = baseClsSegs' * baseClsSegs / M
    CA = baseClsSegs.T @ baseClsSegs / M
    return CA.toarray()
