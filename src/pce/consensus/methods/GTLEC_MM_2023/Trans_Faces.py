import numpy as np

def Trans_Faces(X):
    """
    Transposes each frontal slice of a 3D tensor.
    In Matlab: TX(:,:,i) = X(:,:,i)'
    """
    # [l, m, n] = size(X);
    # TX = zeros(m, l, n);
    # for i = 1 : n
    #     TX(:,:,i)  =  X(:,:,i)';
    # end
    
    # Python equivalent:
    # X shape is (l, m, n) if we follow Matlab convention strictly?
    # NO. Matlab size(X) -> (rows, cols, pages)
    # Python numpy shape -> (rows, cols, pages) usually if constructed that way.
    # However, Python libraries usually prefer (pages, rows, cols) or (depth, height, width).
    # BUT, we must maintain consistency with the caller which likely passes Matlab-style arrays converted to Numpy.
    # The previous code 'TensorEnsemble.py' uses sX = [n, n, 2] and A = np.zeros((n, n, 2)).
    # So the layout is (rows, cols, depth).
    
    # Check dimensions
    if X.ndim != 3:
        raise ValueError("Input must be a 3D tensor")
        
    # Swap axes 0 and 1 (rows and cols)
    TX = np.swapaxes(X, 0, 1)
    
    # Verify:
    # Matlab X(l, m, n) -> TX(m, l, n)
    # Numpy X.shape=(l, m, n) -> swapaxes(0,1) -> (m, l, n). Correct.
    
    return TX
