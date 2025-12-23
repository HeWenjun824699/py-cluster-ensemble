import numpy as np
from scipy.sparse import issparse, csc_matrix

def CDKM(X, label, c):
    """
    Coordinate Descent Method for k-means (Standard Version)
    
    Parameters:
    X (numpy.ndarray or scipy.sparse.spmatrix): d x n data matrix.
    label (numpy.ndarray): n x 1 initial labels.
    c (int): Number of clusters.
    
    Returns:
    tuple: (Y, minO, iter_num, obj)
        Y: Final label vector (n,).
        minO: Converged objective function value.
        iter_num: Number of iterations.
        obj: List of objective function values.
    """
    # Input handling
    if issparse(X):
        d, n = X.shape
        # Ensure efficient column slicing
        X = X.tocsc()
    else:
        d, n = X.shape
        
    # Handle labels (0-based for Python)
    label = np.array(label).flatten().astype(int)
    if label.min() == 1:
        label = label - 1
        
    iter_num = 0
    obj = []
    
    # --- Compute Initial Objective Function Value ---
    sumd_total = 0.0
    for ii in range(c):
        idxi = np.where(label == ii)[0]
        if len(idxi) == 0:
            continue
            
        Xi = X[:, idxi] # d x n_i
        
        # ceni = mean(Xi, 2)
        if issparse(Xi):
            ceni = np.array(Xi.mean(axis=1)).flatten() # (d,)
            # c2 = ceni' * ceni
            c2 = np.dot(ceni, ceni)
            # d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi
            # sum(Xi.^2) -> sum elements of squared matrix down columns
            xi_sq_sum = np.array(Xi.power(2).sum(axis=0)).flatten()
            term3 = 2 * (ceni @ Xi) # vector (n_i,)
        else:
            ceni = np.mean(Xi, axis=1) # (d,)
            c2 = np.dot(ceni, ceni)
            xi_sq_sum = np.sum(Xi**2, axis=0)
            term3 = 2 * np.dot(ceni, Xi)
            
        d2c = xi_sq_sum + c2 - term3
        sumd_total += np.sum(d2c)
        
    obj.append(sumd_total)
    
    # --- Store once XX(i) = ||x_i||^2 ---
    if issparse(X):
        # XX = diag(X'*X) equivalent to sum(X.^2, 1)
        XX = np.array(X.multiply(X).sum(axis=0)).flatten()
    else:
        XX = np.sum(X**2, axis=0)
        
    # --- BB = X * F (Sum of X for each cluster) ---
    BB = np.zeros((d, c))
    for k in range(c):
        idx = np.where(label == k)[0]
        if len(idx) > 0:
            if issparse(X):
                BB[:, k] = np.array(X[:, idx].sum(axis=1)).flatten()
            else:
                BB[:, k] = np.sum(X[:, idx], axis=1)
    
    # --- aa = sum(F, 1) (Cluster counts) ---
    aa = np.zeros(c)
    for k in range(c):
        aa[k] = np.sum(label == k)
        
    # --- FXXF = BB' * BB (Squared norms of cluster sums on diagonal) ---
    # MATLAB: FXXF=BB'*BB;
    # We only track diagonals as the loop only uses FXXF(k,k)
    FXXF_diag = np.sum(BB**2, axis=0) # (c,)
    
    last = np.zeros_like(label) - 1
    
    # --- Main Loop ---
    while np.any(label != last):
        last = label.copy()
        for i in range(n):
            m = label[i]
            if aa[m] == 1:
                continue
                
            # Pre-fetch data for i
            if issparse(X):
                xi = X[:, i].toarray().flatten() # (d,)
            else:
                xi = X[:, i] # (d,)
                
            XX_i = XX[i]
            
            # xi_dot_BB = X(:,i)' * BB
            xi_dot_BB = xi @ BB # (c,)
            
            # Compute Delta
            # MATLAB:
            # if k == m:
            #   V1(k) = FXXF(k,k)- 2 * X(:,i)'* BB(:,k) + XX(i);
            #   delta(k) = FXXF(k,k) / aa(k) - V1(k) / (aa(k) -1);
            # else:
            #   V2(k) =(FXXF(k,k)  + 2 * X(:,i)'* BB(:,k) + XX(i));
            #   delta(k) = V2(k) / (aa(k) +1) -  FXXF(k,k)  / aa(k);
            
            # We calculate this vectorized for all k
            
            # Prepare V2 (assuming k != m logic first)
            V2 = FXXF_diag + 2 * xi_dot_BB + XX_i
            
            # Prepare V1 (only valid for k == m)
            V1_m = FXXF_diag[m] - 2 * xi_dot_BB[m] + XX_i
            
            # Calculate delta for k != m
            # Handle potential division by zero if aa[k] == 0 (though unlikely in std kmeans init)
            with np.errstate(divide='ignore', invalid='ignore'):
                delta = V2 / (aa + 1) - FXXF_diag / aa
            
            # Calculate delta for k == m
            # overwrite delta[m]
            if aa[m] > 1:
                delta[m] = FXXF_diag[m] / aa[m] - V1_m / (aa[m] - 1)
            else:
                delta[m] = -np.inf # Should not happen due to 'continue' check
            
            # Find max delta
            q = np.argmax(delta)
            
            if m != q:
                # Update BB
                BB[:, q] += xi
                BB[:, m] -= xi
                
                # Update aa
                aa[q] += 1
                aa[m] -= 1
                
                # Update FXXF diagonals
                # FXXF(m,m) becomes V1(m) (cost after removal)
                # FXXF(q,q) becomes V2(q) (cost after addition)
                FXXF_diag[m] = V1_m
                FXXF_diag[q] = V2[q]
                
                label[i] = q
        
        iter_num += 1
        
        # --- Compute Objective Function (Re-calc) ---
        sumd_total = 0.0
        for ii in range(c):
            idxi = np.where(label == ii)[0]
            if len(idxi) == 0:
                continue
            Xi = X[:, idxi]
            if issparse(Xi):
                ceni = np.array(Xi.mean(axis=1)).flatten()
                c2 = np.dot(ceni, ceni)
                xi_sq_sum = np.array(Xi.power(2).sum(axis=0)).flatten()
                term3 = 2 * (ceni @ Xi)
            else:
                ceni = np.mean(Xi, axis=1)
                c2 = np.dot(ceni, ceni)
                d2c = np.sum(Xi**2, axis=0) + c2 - 2 * np.dot(ceni, Xi)
                xi_sq_sum = np.sum(Xi**2, axis=0) # Re-eval for consistent var use?
                # Actually d2c calculation above for sparse was:
                # d2c = xi_sq_sum + c2 - term3
                # For dense:
                term3 = 2 * np.dot(ceni, Xi)
            
            # Dense d2c calc
            if not issparse(Xi):
                 d2c = np.sum(Xi**2, axis=0) + c2 - term3
            else:
                 d2c = xi_sq_sum + c2 - term3
                 
            sumd_total += np.sum(d2c)
            
        obj.append(sumd_total)
        
    Y_label = label
    minO = min(obj)
    
    return Y_label, minO, iter_num, np.array(obj)
