import numpy as np
from scipy import sparse
from .util.constructW import constructW
from .util.eig1 import eig1
from .util.discretisation import discretisation

def CEAM(Yi, Wi, c, alpha, k):
    """
    CEAM Algorithm.
    
    Yi: List of indicator matrices (N x K_i)
    Wi: List of weight matrices (N x N)
    c: Number of clusters
    alpha: Parameter
    k: KNN parameter
    """
    m = len(Yi)
    n = Yi[0].shape[0]
    
    options = {}
    options['NeighborMode'] = 'KNN'
    options['WeightMode'] = 'Cosine'
    options['k'] = k
    options['bSelfConnected'] = 1
    
    Di = np.zeros((n, m))
    
    # Pre-process Wi
    for i in range(m):
        # d=1./sqrt(max(sum(Wi{i},2)+m-1,eps));
        # sum(Wi{i}, 2) is sum of rows.
        w_sum = np.array(Wi[i].sum(axis=1)).flatten()
        d = 1.0 / np.sqrt(np.maximum(w_sum + m - 1, np.finfo(float).eps))
        Di[:, i] = d
        
        # WW=bsxfun(@times,Wi{i},d);
        # WW=bsxfun(@times,WW,d');
        # Symmetric scaling: D * W * D
        # In python: elementwise broadcast
        # d is (N,). d[:, None] is (N,1). d[None, :] is (1,N).
        # W is usually sparse.
        
        D_diag = sparse.spdiags(d, 0, n, n)
        Wi[i] = D_diag @ Wi[i] @ D_diag
        
    Ytmp = [[None for _ in range(m)] for _ in range(m)]
    for j in range(m):
        for k_idx in range(m):
             Ytmp[j][k_idx] = Yi[j]
             
    maxiter = 20
    maxiter1 = 5
    
    # Pre-allocate structure for Ytmp updates (to avoid overwriting during calculation)
    # MATLAB: Ytmp2{1,k1}=YY; then Ytmp(j,:)=Ytmp2;
    # So for each j, we compute a new row of Ytmp.
    
    for iter_outer in range(maxiter1):
        for j in range(m):
            Ytmp2 = [None] * m
            
            for i in range(maxiter):
                for k1 in range(m):
                    # YY=alpha.*Wi{k1}*Ytmp{j,k1}+(1-alpha).*Yi{j};
                    term1 = Wi[k1] @ Ytmp[j][k1]
                    YY = alpha * term1 + (1 - alpha) * Yi[j]
                    
                    for k2 in range(m):
                        if k2 == k1:
                            continue
                        else:
                            # dd=alpha.*Di(:,k1).*Di(:,k2);
                            dd = alpha * Di[:, k1] * Di[:, k2]
                            
                            # YY=YY+bsxfun(@times,Ytmp{j,k2},dd);
                            # Ytmp{j,k2} rows are samples. dd matches samples.
                            # bsxfun times with vector column scales rows.
                            
                            if sparse.issparse(Ytmp[j][k2]):
                                # Sparse scaling
                                D_scale = sparse.spdiags(dd, 0, n, n)
                                YY = YY + D_scale @ Ytmp[j][k2]
                            else:
                                YY = YY + Ytmp[j][k2] * dd[:, np.newaxis]
                    
                    Ytmp2[k1] = YY
            
            # Update Ytmp row j
            for k_idx in range(m):
                Ytmp[j][k_idx] = Ytmp2[k_idx]
            
            # Update Wi{j}
            # Wi{j}=constructW(Ytmp{j,j},options);
            # Note: constructW expects dense or sparse matrix. Ytmp{j,j} is likely dense now.
            # Ytmp entries start as sparse (Yi) but become dense due to additions/multiplications?
            # Actually Wi is sparse, Yi is sparse. Result might be dense.
            
            Wi[j] = constructW(Ytmp[j][j], options)
            
            # d=1./sqrt(max(sum(Wi{j},2)+m-1,eps));
            w_sum = np.array(Wi[j].sum(axis=1)).flatten()
            d = 1.0 / np.sqrt(np.maximum(w_sum + m - 1, np.finfo(float).eps))
            
            # WW=bsxfun(@times,Wi{j},d);
            # WW=bsxfun(@times,WW,d');
            D_diag = sparse.spdiags(d, 0, n, n)
            Wi[j] = D_diag @ Wi[j] @ D_diag
            Di[:, j] = d

    # Consensus Graph
    W = Wi[0]
    for t in range(1, m):
        W = W + Wi[t]
        
    W = W / m
    
    # d=1./sqrt(max(sum(W,2),eps));
    w_sum = np.array(W.sum(axis=1)).flatten()
    d = 1.0 / np.sqrt(np.maximum(w_sum, np.finfo(float).eps))
    
    # L=eye(n)-diag(d)*W*diag(d);
    D_diag = sparse.spdiags(d, 0, n, n)
    L = sparse.eye(n) - D_diag @ W @ D_diag
    
    # L=(L+L')./2;
    L = (L + L.T) / 2
    
    # eigvec = eig1(L, c+1, 0);
    eigvec, _, _ = eig1(L, c + 1, 0)
    
    # eigvec(:,1)=[];
    # Remove first eigenvector (corresponding to 0 eigenvalue)
    eigvec = eigvec[:, 1:]
    
    # Y=discretisation(eigvec);
    Y_discrete, _ = discretisation(eigvec)
    
    # [~,pred]=max(Y,[],2);
    # Y is sparse indicator.
    if sparse.issparse(Y_discrete):
        Y_discrete = Y_discrete.toarray()
    pred = np.argmax(Y_discrete, axis=1)
    
    return pred
