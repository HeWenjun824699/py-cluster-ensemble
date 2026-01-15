import numpy as np
from scipy.linalg import eigh, norm

def robust_learn_z(X, lam):
    """
    Learn Z by min_{Z} |X-XZ|_{2,1}+lambda|Z|_F
    Optimized using eigendecomposition of XtX since XtX is constant and symmetric,
    and the other coefficient matrix is diagonal.
    
    Parameters:
    X (numpy.ndarray): data matrix (dim x n)
    lam (float): regularization parameter (lambda)
    
    Returns:
    numpy.ndarray: Z, the similarity matrix
    """
    # print(f"The regularization parameter is: {lam}")
    dim, n = X.shape
    iter_num = 5
    D = np.eye(n)
    XtX = X.T @ X
    
    # Precompute eigendecomposition of XtX
    # XtX is symmetric (semi-definite), so we use eigh
    vals, vecs = eigh(XtX)
    
    # Precompute the RHS for the transformed equation
    # Equation: XtX * Z + Z * diag(temp) = XtX
    # Transformed: diag(vals) * Z_tilde + Z_tilde * diag(temp) = vecs.T @ XtX
    # Since XtX = vecs @ diag(vals) @ vecs.T, 
    # vecs.T @ XtX = diag(vals) @ vecs.T
    
    # vals is (n,), vecs.T is (n, n)
    # Broadcasting vals[:, None] * vecs.T multiplies each row i by vals[i]
    rhs_transformed = vals[:, None] * vecs.T
    
    Z = None
    
    for i in range(iter_num):
        # update Z
        temp_diag = lam * (1.0 / np.diag(D))
        
        # Solving S * Z_tilde + Z_tilde * D_temp = RHS_transformed
        # Element (i, j): vals[i] * Z_tilde[i,j] + Z_tilde[i,j] * temp_diag[j] = RHS_transformed[i,j]
        # Z_tilde[i,j] = RHS_transformed[i,j] / (vals[i] + temp_diag[j])
        
        # Vectorized division
        # Denominator: (n, 1) + (1, n) -> (n, n) matrix of sums
        denom = vals[:, None] + temp_diag[None, :]
        
        z_tilde = rhs_transformed / denom
        
        # Recover Z
        Z = vecs @ z_tilde
        
        # update D
        temp = X @ Z - X
        temp_sq = temp * temp # element-wise square
        
        col_sums = np.sum(temp_sq, axis=0)
        # Avoid division by zero strictly, though data usually prevents it
        d = 0.5 * (1.0 / np.sqrt(col_sums + 1e-10))
        D = np.diag(d)
        
    return Z