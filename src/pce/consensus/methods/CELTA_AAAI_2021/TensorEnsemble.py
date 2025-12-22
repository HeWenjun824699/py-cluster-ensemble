import numpy as np
from .twist.wshrinkObj import wshrinkObj


def TensorEnsemble(F0, W0, lambd):
    """
    Tensor Ensemble optimization.
    
    Args:
        F0: Coherent link matrix
        W0: Coassociation matrix
        lambd: lambda parameter (renamed from lambda to avoid keyword conflict)
        
    Returns:
        A, E, B matrices
    """
    # tol = 1e-8; 
    tol = 1e-8
    # max_iter = 500;
    max_iter = 500
    # rho = 1.1;
    rho = 1.1
    # mu = 1e-4;
    mu = 1e-4
    # max_mu = 1e10;
    max_mu = 1e10
    
    # n=length(F0);
    n = len(F0)
    
    # sX = [n, n, 2];
    sX = [n, n, 2]
    
    # B=zeros(n);
    B = np.zeros((n, n))
    
    # C=zeros(n);
    C = np.zeros((n, n))
    
    # E=B;
    E = B.copy()
    # Lambda1=B;
    Lambda1 = B.copy()
    # Lambda2=B;
    Lambda2 = B.copy()
    # Lambda3=B;
    Lambda3 = B.copy()
    
    # A=zeros(n,n,2);
    A = np.zeros((n, n, 2))
    # T=A;
    T = A.copy()
    
    for iter_idx in range(1, max_iter + 1):
        # update A
        # T(:,:,1)=1*(B-Lambda1/mu);
        T[:,:,0] = 1 * (B - Lambda1 / mu)
        
        # T(:,:,2)=0.5*(W0-E- Lambda2/mu + C- Lambda3/mu);
        T[:,:,1] = 0.5 * (W0 - E - Lambda2 / mu + C - Lambda3 / mu)
        
        ################%%%%%%%
        
        # t = T(:);
        # Matlab's T(:) is column-major (Fortran order). 
        # wshrinkObj likely expects this or we should be consistent.
        # Numpy default is row-major (C order).
        # We should flatten in 'F' order to match Matlab behavior if wshrinkObj depends on it,
        # but usually wshrinkObj operates on the vector and returns a vector which we reshape back.
        t = T.flatten(order='F')
        
        # [a, ~] = wshrinkObj(t,1/mu,sX,0,3)   ;
        # Note: Matlab wshrinkObj returns [a, obj].
        a, _ = wshrinkObj(t, 1/mu, sX, 0, 3)
        
        # Take real part to handle numerical noise from IFFT
        a = a.real
        
        # A= reshape(a, sX);
        # Reshape back using Fortran order to match Matlab's linear indexing
        A = a.reshape(sX, order='F')
        
        ################%%%%%%%
        
        # update E
        # Temp=W0-A(:,:,2)-Lambda2/mu;
        Temp = W0 - A[:,:,1] - Lambda2 / mu
        
        # E=Temp*mu/(2*lambda+mu);
        E = Temp * mu / (2 * lambd + mu)
        
        # update B
        # Temp=A(:,:,1)+Lambda1./mu;
        Temp = A[:,:,0] + Lambda1 / mu
        
        # B=0.5*(Temp+Temp');
        B = 0.5 * (Temp + Temp.T)
        
        # B(F0==1)=1;
        B[F0 == 1] = 1
        
        # B(B<0)=0;
        B[B < 0] = 0
        # B(B>1)=1;
        B[B > 1] = 1
        
        # update C
        # Temp=A(:,:,2)+Lambda3./mu;
        Temp = A[:,:,1] + Lambda3 / mu
        
        # C=0.5*(Temp+Temp');
        C = 0.5 * (Temp + Temp.T)
        
        # C(C<0)=0;
        C[C < 0] = 0
        # C(C>1)=1;
        C[C > 1] = 1
        
        # d1=A(:,:,1)-B;
        d1 = A[:,:,0] - B
        # d2=A(:,:,2)+E-W0;
        d2 = A[:,:,1] + E - W0
        # d3=A(:,:,2)-C;
        d3 = A[:,:,1] - C
        
        print(f'iter: {iter_idx}')
        
        # chg = max([ max(abs(d1(:))),max(abs(d2(:))),max(abs(d3(:)))]);
        chg = max(np.max(np.abs(d1)), np.max(np.abs(d2)), np.max(np.abs(d3)))
        
        if chg < tol:
            break
            
        # Lambda1=Lambda1+mu*d1;
        Lambda1 = Lambda1 + mu * d1
        # Lambda2=Lambda2+mu*d2;
        Lambda2 = Lambda2 + mu * d2
        # Lambda3=Lambda3+mu*d3;
        Lambda3 = Lambda3 + mu * d3
        
        # mu = min(rho*mu,max_mu); 
        mu = min(rho * mu, max_mu)
        
    return A, E, B
