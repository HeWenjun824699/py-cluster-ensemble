import numpy as np

def getHC(X, bound):
    """
    Get High Confidence matrix.
    
    Args:
        X: Input matrix (e.g., CA).
        bound: Threshold.
        
    Returns:
        A: Matrix with values < bound set to 0.
    """
    # E = X; E(X >= bound) = 0; A = X - E;
    # Keeps values >= bound.
    A = np.where(X >= bound, X, 0.0)
    return A
