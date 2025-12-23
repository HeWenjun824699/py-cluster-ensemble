import numpy as np

def solver(H, A, lam):
    """
    Solver for the optimization problem.
    
    Args:
        H: High Confidence Matrix.
        A: LWCA Matrix.
        lam: Lambda parameter.
        
    Returns:
        C: Resulting matrix.
        E: Error matrix.
        t: Number of iterations.
    """
    n = A.shape[0]
    t = 0
    e = 1e-2
    max_iter = 100
    I = np.eye(n)
    C = np.zeros((n, n))
    E = np.zeros((n, n))
    F = np.zeros((n, n))
    r1 = 1.0
    r2 = 1.0
    Y1 = A.copy()
    Y2 = C.copy()
    
    # D = H * ones(n,1) -> row sums
    D = np.sum(H, axis=1)
    
    phi = np.diag(D) - H
    
    # inv_part = (2 * phi + (r1 + r2) * I) \ I
    # Solving X * M = I -> X = inv(M)
    MatToInv = 2 * phi + (r1 + r2) * I
    inv_part = np.linalg.inv(MatToInv)
    
    while t < max_iter:
        t += 1
        
        # update C
        Ct = C.copy()
        P1 = A - E + Y1 / r1
        P2 = F - Y2 / r2
        C = inv_part @ (r1 * P1 + r2 * P2)
        
        # update E
        Et = E.copy()
        E = r1 * (A - C) + Y1
        E = E / (lam + r1)
        E[H > 0] = 0
        
        # update F
        Ft = F.copy()
        F = C + Y2 / r2
        F = (F + F.T) / 2
        F = np.clip(F, 0, 1)
        
        # update Y
        Y1t = Y1.copy()
        residual1 = A - C - E
        Y1 = Y1t + r1 * residual1
        
        Y2t = Y2.copy()
        residual2 = C - F
        Y2 = Y2t + r2 * residual2
        
        # check convergence
        def safe_rel_diff(X, Xt):
            norm_Xt = np.linalg.norm(Xt, 'fro')
            if norm_Xt < 1e-10:
                return 0.0
            return np.abs(np.linalg.norm(X - Xt, 'fro') / norm_Xt)
            
        diffC = safe_rel_diff(C, Ct)
        diffE = safe_rel_diff(E, Et)
        diffF = safe_rel_diff(F, Ft)
        
        # For residuals, diff is norm(residual) / norm(Yt)
        def safe_residual_diff(res, Yt):
            norm_Yt = np.linalg.norm(Yt, 'fro')
            if norm_Yt < 1e-10:
                return 0.0
            return np.abs(np.linalg.norm(res, 'fro') / norm_Yt)
            
        diffY1 = safe_residual_diff(residual1, Y1t)
        diffY2 = safe_residual_diff(residual2, Y2t)
        
        if max([diffC, diffE, diffF, diffY1, diffY2]) < e:
            break
            
    return C, E, t
