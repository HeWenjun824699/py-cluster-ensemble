import numpy as np
import time
from .LibADMM.project_fantope import project_fantope
from .LibADMM.prox_l1 import prox_l1
from .wshrinkObj import wshrinkObj

def TensorEC(H, c, alpha, beta):
    # H: (N, N, V)
    # c: nCluster
    
    N = H.shape[0]
    V = H.shape[2]
    
    # Initialization
    # Cell arrays in MATLAB -> Lists or Dicts in Python
    # S{v} = eye(N)
    S = [np.eye(N) for _ in range(V)]
    Z = [np.eye(N) for _ in range(V)]
    Y1 = [np.zeros((N, N)) for _ in range(V)]
    Y2 = [np.zeros((N, N)) for _ in range(V)]
    A = [np.eye(N) for _ in range(V)]
    
    w = (1/V) * np.ones(V)
    
    Isconverg = 0
    epson = 1e-8
    iter_ = 0
    mu = 10e-6
    max_mu = 10e10
    pho_mu = 3
    rho = 0.01
    max_rho = 10e12
    pho_rho = 3
    
    sX = [N, N, V]
    
    history_objval = []
    
    start_time = time.time()
    
    while Isconverg == 0:
        print(f'----processing iter {iter_ + 1}--------')
        
        for v in range(V):
            # 1 update Z^k and S^k
            # B1 = Z{v} - Y1{v}/mu
            B1 = Z[v] - Y1[v] / mu
            # B2 = A{v} - Y2{v}/rho
            B2 = A[v] - Y2[v] / rho
            
            # B12 = (2 * w(1,v) * H(:,:,v) + mu * B1 + rho * B2)/(2 * w(1,v) + rho + mu);
            # w is 1D array. w[v]
            num = 2 * w[v] * H[:, :, v] + mu * B1 + rho * B2
            den = 2 * w[v] + rho + mu
            B12 = num / den
            
            # B = (B12 + B12')/2;
            B = (B12 + B12.T) / 2
            
            # S{v} = project_fantope(B,c);
            S[v] = project_fantope(B, c)
            
            # Z{v} = prox_l1((S{v}+Y1{v}/mu),alpha/mu);
            Z[v] = prox_l1(S[v] + Y1[v] / mu, alpha / mu)
            
            # Y1{v} =  Y1{v} + mu*(S{v} - Z{v});
            Y1[v] = Y1[v] + mu * (S[v] - Z[v])
            
        # Update G (Tensor optimization)
        # S_tensor = cat(3, S{:,:});
        S_tensor = np.stack(S, axis=2) # (N, N, V)
        Y2_tensor = np.stack(Y2, axis=2)
        
        s_vec = S_tensor.flatten(order='F')
        y2_vec = Y2_tensor.flatten(order='F')
        
        # [a, ~] = wshrinkObj(s+1/rho*y2,beta/rho,sX,0,3);
        # wshrinkObj returns (x, objV)
        a_vec, _ = wshrinkObj(s_vec + (1/rho) * y2_vec, beta/rho, sX, 0, mode=3)
        
        # A_tensor = reshape(a, sX);
        A_tensor = a_vec.reshape(sX, order='F')
        
        # Update omega
        for i in range(V):
            # w(1,i) = 0.5/norm(H(:,:,i)-S{i},'fro');
            diff_norm = np.linalg.norm(H[:, :, i] - S[i], 'fro')
            w[i] = 0.5 / diff_norm if diff_norm > 1e-10 else 1e10 # Avoid division by zero
            
        # Update variables
        # Y2_tensor = Y2_tensor + rho*(S_tensor - A_tensor);
        Y2_tensor = Y2_tensor + rho * (S_tensor - A_tensor)
        
        # Sync back to lists A and Y2 for next iteration
        for v in range(V):
            A[v] = A_tensor[:, :, v]
            Y2[v] = Y2_tensor[:, :, v]
            
        # Converge condition
        Isconverg = 1
        maxvalue = 0
        
        for v in range(V):
            # Check S-A
            norm_S_A = np.max(np.abs(S[v] - A[v])) # inf norm
            if norm_S_A > epson:
                maxvalue = max(maxvalue, norm_S_A)
                Isconverg = 0
                
            # Check S-Z
            norm_S_Z = np.max(np.abs(S[v] - Z[v]))
            if norm_S_Z > epson:
                maxvalue = max(maxvalue, norm_S_Z)
                Isconverg = 0
        
        history_objval.append(maxvalue)
        
        if iter_ > 30:
             Isconverg = 1
             
        iter_ += 1
        mu = min(mu * pho_mu, max_mu)
        rho = min(rho * pho_rho, max_rho)
        
    S0 = np.zeros((N, N))
    for v in range(V):
        S0 = S0 + np.abs(Z[v]) + np.abs(Z[v].T)
        
    # S0 = S0 - diag(diag(S0));
    np.fill_diagonal(S0, 0)
    
    obj = history_objval
    
    return S0, Z, obj
