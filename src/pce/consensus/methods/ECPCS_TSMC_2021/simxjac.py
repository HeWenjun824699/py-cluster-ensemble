import numpy as np

def simxjac(a, b=None):
    """
    Computes extended Jaccard similarity between row objects.
    
    Corresponds to simxjac.m
    
    Parameters:
    -----------
    a : numpy.ndarray
        (n, d) matrix.
    b : numpy.ndarray, optional
        (m, d) matrix. If None, b = a.
        
    Returns:
    --------
    s : numpy.ndarray
        (n, m) similarity matrix.
    """
    if b is None:
        b = a
        
    # a: (n, d), b: (m, d)
    # Matlab logic: temp = a * b';
    temp = np.dot(a, b.T)
    
    # asquared = sum((a.^2),2);
    asquared = np.sum(a**2, axis=1, keepdims=True)
    
    # bsquared = sum((b.^2),2);
    bsquared = np.sum(b**2, axis=1, keepdims=True)
    
    # s = temp ./ ((asquared * ones(1,m)) + (ones(n,1) * bsquared') - temp);
    # Using broadcasting to simulate ones(1,m) and ones(n,1) expansion
    denom = asquared + bsquared.T - temp
    
    # Avoid division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        s = temp / denom
        
    # Replace NaNs (0/0) with 0
    s[np.isnan(s)] = 0.0
    
    return s
