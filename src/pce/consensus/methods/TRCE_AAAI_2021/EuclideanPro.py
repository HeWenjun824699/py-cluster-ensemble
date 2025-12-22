import numpy as np

def EuclideanPro(w, v):
    """
    Euclidean Projection
    min sum(w.*(x-v).^2)
    s.t. sum(x)=1
    
    Args:
        w: weight vector
        v: vector to project
    
    Returns:
        x: projected vector
    """
    
    # w and v should be 1D arrays or flattened
    w = np.squeeze(w)
    v = np.squeeze(v)
    
    w2 = np.sqrt(w)
    c = w2 * v
    a = 1.0 / np.maximum(w2, np.finfo(float).eps)
    s = w * v
    
    maxs = np.max(s)
    theta = maxs
    
    # Boolean indexing for s >= theta
    idx = s >= theta
    s1 = np.sum(idx)
    
    ac = a[idx] * c[idx]
    sumac = np.sum(ac)
    
    aa = a * a
    # suma2 = max(sum(a(idx).^2), eps)
    # Note: indexing should be consistent. 
    # In matlab code: 
    # ac=a(idx).*c(idx); sumac=sum(ac);
    # suma2=max(sum(a(idx).^2),eps);
    suma2 = np.maximum(np.sum(aa[idx]), np.finfo(float).eps)
    
    iter_count = 0
    while True:
        iter_count += 1
        
        theta = (sumac - 1) / suma2
        theta2 = theta - 1e-9
        
        # Update set of active indices
        idx1 = s > theta2
        
        # ac = a .* c (full vector)
        ac_full = a * c
        sumac = np.sum(ac_full[idx1])
        
        sumaa = np.sum(aa[idx1])
        suma2 = np.maximum(sumaa, np.finfo(float).eps)
        
        fihh = -theta * suma2 + sumac - 1
        s2 = np.sum(idx1)
        
        if np.abs(fihh) <= np.finfo(float).eps or s1 == s2:
            break
        
        if iter_count > 100:
            break
        
        s1 = s2
        # idx = idx1 # In matlab this is commented out or implicit?
        # The loop uses idx1 for calculation next time?
        # Matlab code: 
        # idx1=s>theta2; ... if ... break; s1=s2; 
        # Implicitly, the `sumac` and `suma2` are re-calculated using `idx1` at the start of next loop?
        # No, the loop in Matlab:
        # 1. theta = ... (uses sumac/suma2 from PREVIOUS step)
        # 2. idx1 = ...
        # 3. sumac/suma2 updated using idx1
        # 4. check break
        
        # My python loop:
        # 1. theta = ... (uses sumac/suma2 from initialization or previous step)
        # 2. idx1 = ...
        # 3. Update sumac/suma2
        # 4. check break
        # This matches.
    
    x = np.maximum(0, c - theta * a)
    x = x / w2
    
    return x
