import numpy as np

def CoordinateDescent_lyh(alpha, yy, k, n, lam, ind_mat):
    """
    Coordinate Descent for CDEC.
    
    Args:
        alpha (np.ndarray): Weights for base clusterings (T, 1).
        yy (list): Precomputed matrices (T,), can be sparse or dense.
        k (int): Number of clusters.
        n (int): Number of samples.
        lam (float): Lambda parameter.
        ind_mat (np.ndarray): Indicator matrix (n, k).
        
    Returns:
        tuple: (ind_mat, loss_history)
    """
    # Initialize mats
    # mats = zeros(n, n)
    # for i = 1:T: mats += Alpha(i) * YY{i}
    
    mats = np.zeros((n, n))
    t = len(alpha)
    
    for i in range(t):
        if hasattr(yy[i], 'toarray'):
             mats += alpha[i, 0] * yy[i].toarray()
        else:
             mats += alpha[i, 0] * yy[i]
             
    # FF = IndMat' * IndMat -> (K, K) diagonal matrix of cluster sizes
    ff = ind_mat.T @ ind_mat
    mat_d = np.diag(ff).copy() # (K,)
    
    # FHF = IndMat' * mats * IndMat
    fhf = ind_mat.T @ mats @ ind_mat
    mat_a = np.diag(fhf).copy() # (K,)
    
    flag = True
    loss_history = []
    
    while flag:
        flag = False
        # Get current assignments (0-based)
        current_assignments = np.argmax(ind_mat, axis=1)
        
        for i in range(n):
            m = current_assignments[i]
            
            # Check if singleton cluster (avoid emptying it)
            if mat_d[m] == 1:
                continue
                
            max_val = -np.inf
            p = -1 # Target cluster
            
            # Helper for dot products
            mats_i = mats[:, i] 
            dot_vals = ind_mat.T @ mats_i # (K,)
            mats_ii = mats[i, i]
            
            for j in range(k):
                val = 0
                if j == m:
                    denom = mat_d[j] - 1
                    # Logic implies denom > 0 due to 'continue' check above
                    first = mat_a[j] / mat_d[j]
                    second = (mat_a[j] - 2 * dot_vals[j] + mats_ii) / denom
                    
                    val = first - second + lam * (1 - 2 * mat_d[j]) / 2.0
                else:
                    denom = mat_d[j] + 1
                    first = (mat_a[j] + 2 * dot_vals[j] + mats_ii) / denom
                    second = 0
                    if mat_d[j] > 0:
                        second = mat_a[j] / mat_d[j]
                    
                    val = first - second - lam * (1 + 2 * mat_d[j]) / 2.0
                
                if val > max_val:
                    max_val = val
                    p = j
            
            if p != m:
                flag = True
                
                # Update stats
                # matA[m] update
                mat_a[m] = mat_a[m] - 2 * dot_vals[m] + mats_ii
                mat_d[m] = mat_d[m] - 1
                
                # matA[p] update
                mat_a[p] = mat_a[p] + 2 * dot_vals[p] + mats_ii
                mat_d[p] = mat_d[p] + 1
                
                # Update assignment
                ind_mat[i, m] = 0
                ind_mat[i, p] = 1
                current_assignments[i] = p
                
        # Calculate loss
        new_val = 0
        for i in range(k):
            if mat_d[i] > 0:
                new_val += mat_a[i] / mat_d[i]
            new_val -= (lam * (mat_d[i]**2)) / 2.0
            
        loss_history.append(new_val)
        
    return ind_mat, loss_history
