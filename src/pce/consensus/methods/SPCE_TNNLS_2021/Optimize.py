import numpy as np
import sys
import os

from .eig1 import eig1
from .L2_distance_1 import L2_distance_1


def Optimize(Ai, c, gamma):
    """
    Optimize function translated from MATLAB.
    Ai: (n, n, m) numpy array
    c: number of clusters
    gamma: parameter
    """
    
    n, _, m = Ai.shape
    
    # A=sum(Ai,3)./m;
    A = np.sum(Ai, axis=2) / m
    
    # A = (A+A')/2;
    A = (A + A.T) / 2
    
    # D = diag(sum(A));
    D = np.diag(np.sum(A, axis=0))
    
    # L = D - A;
    L = D - A
    
    # [F, ~, evs]=eig1(L, c, 0);
    F, _, evs_vec = eig1(L, c, 0)
    
    # idx_must=find(A==1);
    idx_must = np.where(A == 1)
    # idx_cannot=find(A==0);
    idx_cannot = np.where(A == 0)
    
    S = A.copy()
    
    p = np.ones(m) / m
    phi = np.zeros(m)
    maxiter = 5
    lam = 1.0 # lambda
    gamma = np.sqrt(gamma)
    
    # for qq=0.9:-0.1:0.5
    qq_values = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    for qq in qq_values:
        # rho=2*((qq-1).^2*qq+qq^2*(1-qq))*m^2;
        rho = 2 * ((qq - 1)**2 * qq + qq**2 * (1 - qq)) * m**2
        
        # SS=zeros(n,n);
        SS = np.zeros((n, n))
        for j in range(m):
            # SS=SS+(S-Ai(:,:,j)).^2./p(j);
            SS += (S - Ai[:, :, j])**2 / p[j]
        
        SS = SS * 2
        
        # W=rho./SS;
        with np.errstate(divide='ignore', invalid='ignore'):
            W = rho / SS
        W[np.isnan(W)] = 0
        W[np.isinf(W)] = 0 # Safety, though rho != 0 usually
        
        # W(W>1)=1;
        W[W > 1] = 1
        
        if qq == 0.5:
            maxiter = 20
        
        for ii in range(maxiter):
            B = np.zeros((n, n))
            for j in range(m):
                B += Ai[:, :, j] / p[j]
            
            # distf = L2_distance_1(F',F');
            distf = L2_distance_1(F.T, F.T)
            
            # B=B-lambda.*distf./(2.*W.*W);
            with np.errstate(divide='ignore', invalid='ignore'):
                term = lam * distf / (2 * W * W)
            term[np.isnan(term)] = 0
            term[np.isinf(term)] = 0
            
            B = B - term
            
            # S=B./sum(1./p);
            S = B / np.sum(1 / p)
            
            # S(S<gamma./(W.*sqrt(sum(1./p))))=0;
            with np.errstate(divide='ignore', invalid='ignore'):
                threshold = gamma / (W * np.sqrt(np.sum(1/p)))
            threshold[np.isnan(threshold)] = 0
            threshold[np.isinf(threshold)] = 0 # If W is small?
            
            # If threshold is 0/0 or inf, we need care. 
            # If W is 0 (SS was large), then threshold is inf. S < inf -> S=0. 
            # Matches behavior where similarity should be small if distance is large?
            
            S[S < threshold] = 0
            
            # S(S>1)=1;
            S[S > 1] = 1
            
            # S(idx_must)=1;
            S[idx_must] = 1
            # S(idx_cannot)=0;
            S[idx_cannot] = 0
            
            # S = (S+S')/2;
            S = (S + S.T) / 2
            
            # D = diag(sum(S));
            D = np.diag(np.sum(S, axis=0))
            
            # L = D-S;
            L = D - S
            
            F_old = F
            # [F, ~, ev]=eig1(L, c, 0);
            F, _, ev = eig1(L, c, 0)
            
            for j in range(m):
                # phi(j)=sqrt(sum(sum(((S-Ai(:,:,j)).*W).^2)));
                phi[j] = np.sqrt(np.sum(((S - Ai[:, :, j]) * W)**2))
            
            p = phi / np.sum(phi)
            
            fn1 = np.sum(ev[:c])
            fn2 = np.sum(ev[:c+1])
            
            if fn1 > 1e-9:
                lam = 2 * lam
            elif fn2 < 1e-11:
                lam = lam / 2
                F = F_old
            elif ii > 0: # ii > 1 in MATLAB (1-based)
                break
                
    return S
