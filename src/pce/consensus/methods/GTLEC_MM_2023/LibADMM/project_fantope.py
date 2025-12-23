import numpy as np
from .cappedsimplexprojection import cappedsimplexprojection

def project_fantope(Q, k):
    # [U,D] = eig(Q);
    # In MATLAB, eig(Q) for symmetric Q returns sorted eigenvalues? 
    # Actually MATLAB eig(A) returns eigenvalues not necessarily sorted, but for symmetric A, they are real.
    # But usually we want sorted for processing? 
    # cappedsimplexprojection works on the diagonal D.
    # The order doesn't matter for the projection result on D itself, but U must match D.
    
    # np.linalg.eigh returns eigenvalues in ascending order.
    d, U = np.linalg.eigh(Q)
    
    # Dr = cappedsimplexprojection(diag(D),k);
    Dr = cappedsimplexprojection(d, k)
    
    # X = U*diag(Dr)*U';
    X = U @ np.diag(Dr) @ U.T
    
    return X
