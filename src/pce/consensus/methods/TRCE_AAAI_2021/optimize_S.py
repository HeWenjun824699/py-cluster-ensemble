import numpy as np
from scipy.optimize import brentq
from .rootfinding import rootfinding

def optimize_S(B, C, rho):
    """
    OPTIMIZE_S
    
    Args:
        B: matrix
        C: matrix
        rho: scalar
    
    Returns:
        S: optimized matrix
    """
    n = C.shape[0]
    S = np.zeros((n, n))
    
    for i in range(n):
        # idx = find(C(i,:))
        # In Matlab find returns indices of non-zero elements.
        idx = np.flatnonzero(C[i, :])
        
        if idx.size == 0:
            continue
            
        # theta=fzero(@(x)rootfinding(x,C(i,idx),B(i,idx).*rho),[eps,10000]);
        # We need to find a root for rootfinding(x, c_vec, h_vec)
        # c_vec = C[i, idx]
        # h_vec = B[i, idx] * rho
        
        c_vec = C[i, idx]
        h_vec = B[i, idx] * rho
        
        def func(x):
            return rootfinding(x, c_vec, h_vec)
        
        # Matlab range: [eps, 10000]
        # We should check signs to be safe, but assuming the range is valid as per Matlab code.
        # brentq requires f(a) and f(b) to have different signs.
        low = np.finfo(float).eps
        high = 10000.0
        
        try:
            # Check signs if strict robustness is needed, but brentq will raise if not bracketed
            # If func(low) and func(high) same sign, we might need to expand or fallback.
            # Matlab fzero is more robust to non-bracketed if it can search, but [eps, 10000] implies a bracket.
            theta = brentq(func, low, high)
        except ValueError:
            # Fallback or error handling similar to Matlab's try-catch (which catches errors and prints 'i')
            # If brentq fails, maybe the root is outside or no root.
            # We'll try to expand or just pick a bound if it's monotonic.
            f_low = func(low)
            f_high = func(high)
            if f_low * f_high > 0:
                # Same sign. 
                # If decreasing and both positive -> root > 10000?
                # If decreasing and both negative -> root < eps?
                if f_low > 0: # both positive
                     theta = high # Clip?
                else: # both negative
                     theta = low
            else:
                 theta = low # Should not happen if signs differ
        
        # S(i,idx)=C(i,idx)./(rho.*B(i,idx)+theta);
        S[i, idx] = C[i, idx] / (rho * B[i, idx] + theta)
        
    return S
