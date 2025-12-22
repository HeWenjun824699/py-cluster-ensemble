import numpy as np
from .discretisationEigenVectorData import discretisationEigenVectorData

def discretisation(EigenVectors):
    """
    Discretizes continuous Ncut vectors.
    
    Corresponds to discretisation.m
    
    Parameters:
    -----------
    EigenVectors : numpy.ndarray
        (n, k) matrix of continuous Ncut vectors.
        
    Returns:
    --------
    EigenvectorsDiscrete : scipy.sparse.csr_matrix
        (n, k) discrete Ncut vectors (indicator matrix).
    EigenVectors : numpy.ndarray
        Normalized EigenVectors.
    """
    EigenVectors = EigenVectors.copy()
    n, k = EigenVectors.shape
    
    # vm = sqrt(sum(EigenVectors.*EigenVectors,2));
    vm = np.sqrt(np.sum(EigenVectors**2, axis=1, keepdims=True))
    
    # EigenVectors = EigenVectors./repmat(vm,1,k);
    # Handle division by zero if any row is all zeros (unlikely for eigenvectors but good practice)
    with np.errstate(divide='ignore', invalid='ignore'):
        EigenVectors = EigenVectors / vm
    
    EigenVectors[np.isnan(EigenVectors)] = 0.0
    
    # R=zeros(k);
    R = np.zeros((k, k))
    
    # R(:,1)=EigenVectors(1+round(rand(1)*(n-1)),:)';
    # Matlab: 1 + round(rand * (n-1)). Indices are 1-based.
    # Python: random integer between 0 and n-1.
    # Using specific logic to mimic: round(rand * (n-1)) -> 0 to n-1.
    rand_idx = int(np.round(np.random.rand() * (n - 1)))
    R[:, 0] = EigenVectors[rand_idx, :]
    
    # c=zeros(n,1);
    c = np.zeros(n)
    
    for j in range(1, k): # j from 1 to k-1 (2 to k in Matlab)
        # c=c+abs(EigenVectors*R(:,j-1));
        # R(:,j-1) is column vector.
        # np.dot(EigenVectors, R[:, j-1]) gives vector of size n.
        c = c + np.abs(np.dot(EigenVectors, R[:, j-1]))
        
        # [minimum,i]=min(c);
        i = np.argmin(c)
        
        # R(:,j)=EigenVectors(i,:)';
        R[:, j] = EigenVectors[i, :]
        
    lastObjectiveValue = 0.0
    exitLoop = 0
    nbIterationsDiscretisation = 0
    nbIterationsDiscretisationMax = 20
    
    EigenvectorsDiscrete = None
    
    while exitLoop == 0:
        nbIterationsDiscretisation += 1
        
        # EigenvectorsDiscrete = discretisationEigenVectorData(EigenVectors*R);
        # Rotated vectors = EigenVectors @ R
        rotated_vecs = np.dot(EigenVectors, R)
        EigenvectorsDiscrete = discretisationEigenVectorData(rotated_vecs)
        
        # [U,S,V] = svd(EigenvectorsDiscrete'*EigenVectors,0);
        # EigenvectorsDiscrete is sparse. Convert to dense for dot product or use sparse dot.
        # EigenvectorsDiscrete.T @ EigenVectors
        # SVD of k x k matrix.
        mat_for_svd = EigenvectorsDiscrete.T.dot(EigenVectors) # .dot handles sparse
        
        # Matlab svd(A,0) -> U, S, V. A = U * S * V'.
        # Python np.linalg.svd(A) -> U, S, Vt. A = U * diag(S) * Vt.
        # Vt in Python is V' in Matlab.
        U, S, Vt = np.linalg.svd(mat_for_svd, full_matrices=False)
        
        # NcutValue=2*(n-trace(S));
        # S is 1D array of singular values in numpy
        NcutValue = 2 * (n - np.sum(S))
        
        # if abs(NcutValue-lastObjectiveValue) < eps | nbIterationsDiscretisation > nbIterationsDiscretisationMax
        if (np.abs(NcutValue - lastObjectiveValue) < np.finfo(float).eps) or (nbIterationsDiscretisation > nbIterationsDiscretisationMax):
            exitLoop = 1
        else:
            lastObjectiveValue = NcutValue
            # R=V*U';
            # V in Matlab is Vt.T in Python.
            # So R = Vt.T @ U.T = (U @ Vt).T ?
            # Wait. Matlab: [U, S, V] = svd(A). A = U*S*V'.
            # R = V * U'.
            # Python: U, S, Vt = svd(A). A = U*S*Vt.
            # So V_matlab = Vt.T.
            # R = Vt.T @ U.T
            R = np.dot(Vt.T, U.T)
            
    return EigenvectorsDiscrete, EigenVectors
