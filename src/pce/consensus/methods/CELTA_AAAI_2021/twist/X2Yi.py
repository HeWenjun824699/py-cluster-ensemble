import numpy as np
from .Trans_Faces import Trans_Faces

def shiftdim(X, n):
    """
    Mimics Matlab's shiftdim(X, n).
    Shifts the dimensions of X by n.
    When n > 0, shiftdim shifts the dimensions to the left and wraps the n leading dimensions to the end.
    Example: size(X) = [4 2 3], shiftdim(X, 1) -> size is [2 3 4]
    
    Note: Python numpy.rollaxis or moveaxis can do this.
    rollaxis(a, axis, start=0) rolls the specified axis to a given position.
    
    Matlab shiftdim(X, 1): [d1, d2, d3] -> [d2, d3, d1]
    Matlab shiftdim(X, 2): [d1, d2, d3] -> [d3, d1, d2]
    
    In Numpy:
    np.rollaxis(X, 0, 3) is equivalent to shiftdim(X, 1) for 3D array?
    rollaxis(X, 0, 3) moves axis 0 to position 3.
    Indices: 0, 1, 2. Move 0 to 3 (end): 1, 2, 0. Correct.
    """
    if n == 0:
        return X
    
    # Handle strictly 3D case as per project usage
    ndim = X.ndim
    n = n % ndim # Handle wrapping if n >= ndim, though Matlab behaves differently for n > ndim (shifts into singletons)
    
    # For this project, we assume standard rotation.
    # shiftdim(X, 1) moves 0th dim to end.
    # shiftdim(X, 2) moves 0th and 1st dim to end.
    
    # We can use np.rollaxis. 
    # To shift 1 (d1 d2 d3 -> d2 d3 d1): move axis 0 to end.
    if n == 1:
        return np.moveaxis(X, 0, -1)
    elif n == 2:
        return np.moveaxis(X, [0, 1], [-2, -1]) # Move 0,1 to end?
        # d1 d2 d3 -> d3 d1 d2.
        # moveaxis(X, 0, -1) -> d2 d3 d1
        # moveaxis(..., 0, -1) again -> d3 d1 d2
        # Or just: np.roll(X, -n) on axes? No.
        
        # Let's visualize:
        # np.moveaxis(X, source, destination)
        # shiftdim(X, 2): [0, 1, 2] -> [2, 0, 1]
        # We want axis 2 to become axis 0, axis 0 to become axis 1, axis 1 to become axis 2.
        # equivalent to moving axis 2 to position 0.
        return np.moveaxis(X, 2, 0) # [2, 0, 1]
    
    # General logic for positive n (rotate left)
    # [0, 1, 2, ..., k] -> [n, n+1, ..., k, 0, ..., n-1]
    axes = list(range(ndim))
    new_axes = axes[n:] + axes[:n]
    return np.transpose(X, new_axes)

def X2Yi(X, i):
    """
    function Y = X2Yi(X, i)
    Y = Trans_Faces(shiftdim(X, i-1));
    """
    # Y = Trans_Faces(shiftdim(X, i-1));
    # Note: Matlab passes 1-based index 'i'.
    
    shifted_X = shiftdim(X, i - 1)
    Y = Trans_Faces(shifted_X)
    
    return Y
