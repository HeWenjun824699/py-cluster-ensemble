import numpy as np
from scipy import sparse


def norml(x, mode):
    """
    Row normalization of matrix x.
    
    MATLAB equivalent:
    function x = norml(x,mode)
    """
    
    if mode == 0:
        return x
        
    elif mode == 1:
        # x = diag(sparse(1./sum(abs(x),2))) * x;
        if sparse.issparse(x):
            row_sums = np.array(np.sum(np.abs(x), axis=1)).flatten()
        else:
            row_sums = np.sum(np.abs(x), axis=1)
            
        with np.errstate(divide='ignore'):
            inv_sums = 1.0 / row_sums
        inv_sums[np.isinf(inv_sums)] = 0
        inv_sums[np.isnan(inv_sums)] = 0
        
        if sparse.issparse(x):
            D = sparse.diags(inv_sums)
            x = D @ x
        else:
            x = x * inv_sums[:, np.newaxis]
            
    elif mode == 2:
        # x = diag(sparse(1./sum((x.^2),2).^(1/2))) * x;
        if sparse.issparse(x):
            # For sparse matrices, power is efficient
            row_sq_sums = np.array(np.sum(x.power(2), axis=1)).flatten()
        else:
            row_sq_sums = np.sum(x**2, axis=1)
            
        row_norms = np.sqrt(row_sq_sums)
        
        with np.errstate(divide='ignore'):
            inv_norms = 1.0 / row_norms
        inv_norms[np.isinf(inv_norms)] = 0
        inv_norms[np.isnan(inv_norms)] = 0

        if sparse.issparse(x):
            D = sparse.diags(inv_norms)
            x = D @ x
        else:
            x = x * inv_norms[:, np.newaxis]
            
    else:
        print('norml: mode not supported')
        
    return x
