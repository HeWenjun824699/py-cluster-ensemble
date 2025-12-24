import numpy as np
from scipy import sparse
from .discretisationEigenVectorData import discretisationEigenVectorData

def discretisation(EigenVectors):
    """
    Discretizes continuous Ncut vector.
    """
    n, k = EigenVectors.shape
    
    # vm = max(sqrt(sum(EigenVectors.*EigenVectors,2)),eps);
    vm = np.sqrt(np.sum(EigenVectors * EigenVectors, axis=1))
    vm = np.maximum(vm, np.finfo(float).eps)
    
    # EigenVectors = EigenVectors./repmat(vm,1,k);
    EigenVectors = EigenVectors / vm[:, np.newaxis]
    
    # R=zeros(k);
    R = np.zeros((k, k))
    
    # R(:,1)=EigenVectors(1+round(rand(1)*(n-1)),:)';
    # MATLAB: 1+round(rand(1)*(n-1)) -> random index from 1 to n.
    # Python: random index from 0 to n-1.
    rand_idx = int(np.round(np.random.rand() * (n - 1)))
    R[:, 0] = EigenVectors[rand_idx, :].T
    
    # c=zeros(n,1);
    c = np.zeros(n)
    
    for j in range(1, k):
        # c=c+abs(EigenVectors*R(:,j-1));
        c = c + np.abs(EigenVectors @ R[:, j-1])
        
        # [minimum,i]=min(c);
        i = np.argmin(c)
        
        # R(:,j)=EigenVectors(i,:)';
        R[:, j] = EigenVectors[i, :].T
        
    lastObjectiveValue = 0
    exitLoop = 0
    nbIterationsDiscretisation = 0
    nbIterationsDiscretisationMax = 20
    
    while exitLoop == 0:
        nbIterationsDiscretisation += 1
        
        # EigenvectorsDiscrete = discretisationEigenVectorData(EigenVectors*R);
        EigenvectorsDiscrete = discretisationEigenVectorData(EigenVectors @ R)
        
        # [U,S,V] = svd(EigenvectorsDiscrete'*EigenVectors,0);
        # MATLAB svd(A, 0) is economy size.
        # Python: full_matrices=False
        # MATLAB: [U,S,V] = svd(A) => A = U*S*V'
        # Python: u, s, vh = svd(A) => A = u*diag(s)*vh
        # So MATLAB U = Python u
        # MATLAB S = Python s (diagonal)
        # MATLAB V = Python vh.T
        
        target = EigenvectorsDiscrete.T @ EigenVectors
        if sparse.issparse(target):
            target = target.toarray()
            
        u, s, vh = np.linalg.svd(target, full_matrices=False)
        
        # NcutValue=2*(n-trace(S));
        NcutValue = 2 * (n - np.sum(s))
        
        if abs(NcutValue - lastObjectiveValue) < np.finfo(float).eps or nbIterationsDiscretisation > nbIterationsDiscretisationMax:
            exitLoop = 1
        else:
            lastObjectiveValue = NcutValue
            # R=V*U';
            # MATLAB: V*U'
            # Python: vh.T @ u.T
            R = vh.T @ u.T
            
    return EigenvectorsDiscrete, EigenVectors
