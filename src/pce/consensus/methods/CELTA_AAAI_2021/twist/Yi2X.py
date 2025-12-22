import numpy as np
from .Trans_Faces import Trans_Faces

def shiftdim(X, n):
    """
    Mimics Matlab's shiftdim(X, n).
    See X2Yi.py for implementation details.
    """
    if n == 0:
        return X
    ndim = X.ndim
    n = n % ndim
    axes = list(range(ndim))
    new_axes = axes[n:] + axes[:n]
    return np.transpose(X, new_axes)

def Yi2X(Y, i):
    """
    function X = Yi2X(Y, i)
    if i == 3
        X = shiftdim(Trans_Faces(Y),i+1);
    elseif i == 2
        X = shiftdim(Trans_Faces(Y),i);
    else
        X = shiftdim(Trans_Faces(Y),i-1);
    end
    end
    """
    tf_Y = Trans_Faces(Y)
    
    if i == 3:
        # X = shiftdim(Trans_Faces(Y),i+1);
        X = shiftdim(tf_Y, i + 1)
    elif i == 2:
        X = shiftdim(tf_Y, i)
    else:
        X = shiftdim(tf_Y, i - 1)
        
    return X
