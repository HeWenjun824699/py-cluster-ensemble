import numpy as np


def simxjac(a, b=None):
    """
    Computes extended Jaccard similarity between row objects in matrices a and b.
    
    MATLAB equivalent:
    function s = simxjac(a,b)
    """
    if b is None:
        b = a
    
    n = a.shape[0]
    m = b.shape[0]
    d = a.shape[1]
    
    if d != b.shape[1]:
        print('simxjac: data dimensions do not match')
        return None

    # temp = real(a) * b'
    temp = np.real(a) @ b.T
    
    # asquared = sum((a.^2),2); 
    asquared = np.sum(a**2, axis=1).reshape(-1, 1)
    
    # bsquared = sum((b.^2),2); 
    bsquared = np.sum(b**2, axis=1).reshape(-1, 1)
    
    # s = temp ./ ((asquared * ones(1,m)) + (ones(n,1) * bsquared') - temp);
    # Broadcasting handles the ones(...) expansion
    denominator = asquared + bsquared.T - temp
    
    # Handle division by zero or small numbers if necessary? 
    # MATLAB behavior is followed here (standard division)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = temp / denominator
    
    # MATLAB doesn't zero out NaNs explicitly in the source provided, 
    # but often Jaccard implies 0 if denominator is 0 (0/0).
    # However, simxjac formula: 0/0 -> NaN.
    
    return s
