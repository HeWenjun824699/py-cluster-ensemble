import numpy as np
from scipy.sparse import csc_matrix, diags, eye
from scipy.optimize import minimize

from .CoordinateDescent_lyh import CoordinateDescent_lyh

def CDEC_lyh(T, K, BPi, lam, gamma, now):
    """
    CDEC algorithm main driver.
    
    Args:
        T (int): Number of base clusterings.
        K (int): Number of clusters.
        BPi (np.ndarray): Base partitions (n_samples, T).
        lam (float): Lambda parameter.
        gamma (float): Gamma parameter.
        now (np.ndarray): Initial clustering labels (n_samples,).
                          
    Returns:
        np.ndarray: Final labels.
    """
    n = BPi.shape[0]
    
    # Initialize Alpha
    alpha = np.ones((T, 1)) / T
    
    # Base clusters processing
    # Construct YY matrices
    YY = []
    
    for i in range(T):
        labels = BPi[:, i]
        # Handle 1-based indexing if present
        if np.min(labels) == 1:
             col = labels.astype(int) - 1
        else:
             col = labels.astype(int)
        
        row = np.arange(n)
        # Ensure correct dimension
        n_c = np.max(col) + 1
        data = np.ones(n)
        
        bc = csc_matrix((data, (row, col)), shape=(n, n_c))
        
        # YY_inv = baseCluster{i}' * baseCluster{i} -> Diagonal matrix of cluster sizes
        yy_inv = bc.T @ bc
        sizes = np.array(yy_inv.diagonal())
        
        # Avoid div by zero
        sizes[sizes == 0] = 1 
        inv_diag_vals = 1.0 / sizes
        
        d_mat = diags(inv_diag_vals)
        
        # YY{i} = baseCluster{i} * inv(diag) * baseCluster{i}'
        term = bc @ d_mat @ bc.T
        YY.append(term)

    # Main loop parameters
    max_iter = 50
    times = 0
    now_val = np.inf
    loss_values = []
    
    # IndMat init
    ind_mat = np.zeros((n, K))
    
    # Handle 'now' (initial labels)
    current_labels = now.astype(int)
    if np.min(current_labels) == 1:
        current_labels -= 1
        
    ind_mat[np.arange(n), current_labels] = 1
    
    while times < max_iter:
        # Step 1: Update IndMat
        ind_mat, loss = CoordinateDescent_lyh(alpha, YY, K, n, lam, ind_mat)
        
        # Compute objective value for convergence check
        hh_mat = sum(alpha[i, 0] * YY[i] for i in range(T))
        
        if hasattr(hh_mat, 'toarray'):
             t_val = (hh_mat @ hh_mat).trace()
        else:
             t_val = np.trace(hh_mat @ hh_mat)
             
        last_loss = loss[-1]
        
        sum_val = t_val - 2 * last_loss + K + gamma * (alpha.T @ alpha)[0, 0]
        loss_values.append(sum_val)
        
        if abs(now_val - sum_val) < 1e-5:
            break
            
        now_val = sum_val
        times += 1
        
        # Step 2: Update Alpha
        alpha = get_alpha(ind_mat, T, YY, gamma)
        
    # Final label extraction
    label_final = np.argmax(ind_mat, axis=1)
    
    # If the ecosystem expects 1-based labels (like MATLAB), we might need to add 1.
    # But for Python libraries, 0-based is standard. 
    # Returning 0-based.
    
    return label_final

def get_alpha(ind_mat, T, YY, gamma):
    """
    Solve quadratic programming for Alpha.
    min alpha.T * A * alpha - 2 * B.T * alpha
    st sum(alpha)=1, alpha>=0
    """
    # Matrix A
    A = np.zeros((T, T))
    
    # Optimize trace calculation: trace(A B) = sum(A .* B) if symmetric
    for i in range(T):
        for j in range(i, T):
            # Element-wise multiply and sum
            val = (YY[i].multiply(YY[j])).sum()
            A[i, j] = val
            A[j, i] = val
            
    A = A + gamma * np.eye(T)
    
    # Vector B
    B = np.zeros((T, 1))
    
    # FF term logic:
    # B[i] = trace(YY[i] * IndMat * inv(IndMat'*IndMat) * IndMat')
    #      = trace(IndMat' * YY[i] * IndMat * inv(IndMat'*IndMat))
    
    cluster_sizes = ind_mat.sum(axis=0)
    inv_sizes = np.zeros_like(cluster_sizes)
    inv_sizes[cluster_sizes > 0] = 1.0 / cluster_sizes[cluster_sizes > 0]
    
    for i in range(T):
         # M = IndMat.T @ YY[i] @ IndMat -> (K, K)
         m_mat = ind_mat.T @ YY[i] @ ind_mat
         if hasattr(m_mat, 'diagonal'):
             m_diag = m_mat.diagonal()
         else:
             m_diag = np.diag(m_mat)
             
         val = np.sum(m_diag * inv_sizes)
         B[i, 0] = val
         
    # Solve QP using scipy.optimize.minimize
    # Objective: x' A x - 2 B' x
    # Factor 2 can be pulled out or kept. MATLAB: quadprog(2*A, -2*B, ...) -> min 1/2 x'(2A)x + (-2B)'x = x'Ax - 2B'x
    
    def objective(x):
        return x.T @ A @ x - 2 * B.flatten().dot(x)
        
    def jac(x):
        return 2 * A @ x - 2 * B.flatten()
        
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    ]
    bounds = [(0, None) for _ in range(T)]
    
    x0 = np.ones(T) / T
    
    # method='SLSQP' handles constraints and bounds well
    res = minimize(objective, x0, method='SLSQP', jac=jac, bounds=bounds, constraints=constraints)
    
    new_alpha = res.x.reshape((T, 1))
    return new_alpha
