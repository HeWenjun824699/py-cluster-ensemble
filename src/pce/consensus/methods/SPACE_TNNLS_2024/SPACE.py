import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .alg.L2_distance_1 import l2_distance_1
from .alg.expand import expand
from .alg.selectPairs import select_pairs
from .util.eig1 import eig1


def SPACE(Ai, c, gamma, y, batchsize, T, delta):
    """
    SPACE_NEW Modified version of SPACE without internal evaluation.
    
    Parameters:
    Ai : numpy.ndarray
        (n, n, m) array of base clusterings (similarity matrices).
    c : int
        Number of clusters.
    gamma : float
    y : numpy.ndarray
        Ground truth labels (for constraint generation).
    batchsize : int
    T : int
        Number of iterations.
    delta : float
    
    Returns:
    S : numpy.ndarray
        Consensus similarity matrix.
    constraints : numpy.ndarray
        Generated constraints.
    """
    n, _, m = Ai.shape
    
    # A=sum(Ai,3)./m;
    A = np.sum(Ai, axis=2) / m
    
    # A = (A+A')/2;
    A = (A + A.T) / 2
    
    # D = diag(sum(A)); L = D - A;
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    
    # [F, ~, evs]=eig1(L, c, 0);
    F, _, evs = eig1(L, c, 0)
    
    S = A.copy()
    
    p = np.ones(m) / m
    phi = np.zeros(m)
    maxiter = 5
    lam = 1 # lambda is reserved keyword
    
    gamma = np.sqrt(gamma)
    cons = np.empty((0, 3))
    Y_cannot = np.empty((n, 0)) # Will grow columns
    cannot_index = np.empty((0, 2), dtype=int)
    idx_labeled = [] # List of indices
    qq = 0.9
    constraints = np.empty((0,), dtype=int) # Flattened indices
    
    # dia=1:n+1:n*n; (Diagonal indices)
    # In Python, we can just use np.fill_diagonal or S[np.arange(n), np.arange(n)]
    
    for iter_idx in range(T):
        rho = 2 * ((qq - 1)**2 * qq + qq**2 * (1 - qq)) / m
        SS = np.zeros((n, n))
        
        for j in range(m):
            SS = SS + (S - Ai[:, :, j])**2 * p[j]**2
            
        SS = SS * 2
        
        # W=rho./SS; W(W>1)=1;
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            W = rho / SS
        W[np.isnan(W)] = 1 # Assume 0/0 is 1? Or Inf? MATLAB behaves specifically.
        # If SS is 0, W is Inf. W(W>1)=1 handles Inf.
        W[W > 1] = 1
        
        WW = W.copy()
        
        # WW(idx_labeled)=2;
        if len(idx_labeled) > 0:
            # idx_labeled are linear indices. 
            # In Python, assignment to flat array or unraveled
            WW.flat[idx_labeled] = 2
            
        # WW=tril(WW)+triu(ones(n,n).*2);
        WW = np.tril(WW) + np.triu(np.ones((n, n)) * 2, 1) # k=1 for strict upper tri
        
        SSS = np.zeros((n, n))
        
        # labeled=selectPairs(WW,S,batchsize,n);
        labeled = select_pairs(WW, S, batchsize, n)
        
        del WW
        
        # W(idx_labeled)=1;
        if len(idx_labeled) > 0:
            W.flat[idx_labeled] = 1
            
        # constraints=[constraints;labeled];
        constraints = np.concatenate([constraints, labeled])
        
        # [i_label,j_label]=ind2sub([n,n],labeled);
        i_label, j_label = np.unravel_index(labeled, (n, n))
        
        for kk in range(batchsize):
            # Safe check if batchsize > available labeled
            if kk >= len(i_label):
                break
                
            if y[i_label[kk]] == y[j_label[kk]]:
                # cons=[cons;[i_label(kk),j_label(kk),1]];
                new_row = np.array([[i_label[kk], j_label[kk], 1]])
                cons = np.vstack([cons, new_row])
            else:
                new_row = np.array([[i_label[kk], j_label[kk], -1]])
                cons = np.vstack([cons, new_row])
                
                # Y_cannot=[Y_cannot,zeros(n,1)];
                Y_cannot = np.hstack([Y_cannot, np.zeros((n, 1))])
                
                # Y_cannot(i_label(kk),end)=1;
                Y_cannot[i_label[kk], -1] = 1
                Y_cannot[j_label[kk], -1] = -1
                
                # cannot_index=[cannot_index;[i_label(kk),j_label(kk)]];
                cannot_index = np.vstack([cannot_index, np.array([[i_label[kk], j_label[kk]]])])
                
        # [V,E,~]=expand(cons);
        # We need V_old and cannot logic if we want to persist, but SPACE_new.m 
        # calls expand(cons) without V_old inside the loop?
        # Re-reading SPACE_new.m:
        # cons is accumulating. 
        # [V,E,~]=expand(cons); -> It passes the accumulated 'cons'.
        # expand.m handles cons.
        
        V, E, _ = expand(cons)
        
        # for jj=1:length(V) SSS(V{jj},V{jj})=1; end
        for jj in range(len(V)):
            # V[jj] is a set of indices
            indices = list(V[jj])
            # SSS[indices, indices] = 1 -> this doesn't work directly for block assignment in numpy like MATLAB
            # We need explicit indexing
            ixgrid = np.ix_(indices, indices)
            SSS[ixgrid] = 1
            
        # [i_E,j_E]=find(E>0);
        i_E, j_E = np.where(E > 0)
        
        for jj in range(len(i_E)):
            # SSS(V{i_E(jj)},V{j_E(jj)})=-1;
            r_indices = list(V[i_E[jj]])
            c_indices = list(V[j_E[jj]])
            ixgrid = np.ix_(r_indices, c_indices)
            SSS[ixgrid] = -1
            
        # idx_labeled=find(SSS(:)~=0);
        idx_labeled = np.where(SSS.flatten() != 0)[0]
        
        # idx_must=find(SSS(:)==1);
        idx_must = np.where(SSS.flatten() == 1)[0]
        
        # idx_cannot=find(SSS(:)==-1);
        idx_cannot = np.where(SSS.flatten() == -1)[0]
        
        # [~,num_can]=size(Y_cannot);
        num_can = Y_cannot.shape[1]
        
        for ii in range(maxiter):
            B = np.zeros((n, n))
            for j in range(m):
                B = B + Ai[:, :, j] * p[j]**2
                
            distf = l2_distance_1(F.T, F.T)
            distY = l2_distance_1(Y_cannot.T, Y_cannot.T)
            
            beta = 1 * lam
            
            if distY.size == 0 or (distY.shape[0] == 1 and distY.shape[1] == 0): # Check if empty
                 # B=B-(lambda.*distf)./(2.*W.*W);
                 B = B - (lam * distf) / (2 * W * W)
            else:
                 B = B - (lam * distf + beta * distY) / (2 * W * W)
                 
            # S=B./sum(p.^2);
            S = B / np.sum(p**2)
            
            # S(S<gamma./(W.*sqrt(sum(p.^2))))=0;
            threshold_S = gamma / (W * np.sqrt(np.sum(p**2)))
            S[S < threshold_S] = 0
            
            S[S > 1] = 1
            
            # S(idx_must)=1;
            S.flat[idx_must] = 1
            
            # S(idx_cannot)=0;
            S.flat[idx_cannot] = 0
            
            # S(dia)=1;
            np.fill_diagonal(S, 1)
            
            S = (S + S.T) / 2
            D = np.diag(np.sum(S, axis=1))
            L = D - S
            
            F_old = F.copy()
            
            F, _, ev = eig1(L, c, 0)
            
            # evs(:,ii+1) = ev; 
            # Not used in return?
            
            # [~, ypred]=graphconncomp(sparse(S));
            # scipy.sparse.csgraph.connected_components returns (n_components, labels)
            # labels is 0-based
            n_components, ypred = csgraph.connected_components(sparse.csr_matrix(S), directed=False)
            # Note: MATLAB graphconncomp returns 1-based labels usually, but consistency matters most.
            
            for j in range(num_can):
                l_idx = cannot_index[j, :] # [i, j]
                
                # Check labels
                if ypred[l_idx[0]] != ypred[l_idx[1]]:
                    yk = np.zeros(n)
                    # idx_1=find(ypred==ypred(l_idx(1)));
                    idx_1 = np.where(ypred == ypred[l_idx[0]])[0]
                    idx_2 = np.where(ypred == ypred[l_idx[1]])[0]
                    
                    yk[idx_1] = 1
                    yk[idx_2] = -1
                    Y_cannot[:, j] = yk
                else:
                    # idx_3=find(ypred==ypred(l_idx(1)));
                    idx_3 = np.where(ypred == ypred[l_idx[0]])[0]
                    
                    # u_idx=setdiff(idx_3,l_idx);
                    u_idx = np.setdiff1d(idx_3, l_idx)
                    
                    if len(u_idx) > 0:
                        # Lul=L(u_idx,l_idx);
                        Lul = L[np.ix_(u_idx, l_idx)]
                        
                        # Luu=L(u_idx,u_idx);
                        Luu = L[np.ix_(u_idx, u_idx)]
                        
                        # l1=Lul(:,1)-Lul(:,2);
                        l1 = Lul[:, 0] - Lul[:, 1]
                        
                        # ykk=(Luu)\l1;
                        # Use lstsq or solve
                        try:
                            ykk = np.linalg.solve(Luu, l1)
                        except np.linalg.LinAlgError:
                            ykk = np.linalg.lstsq(Luu, l1, rcond=None)[0]
                            
                        # Y_cannot(u_idx,j)=ykk;
                        Y_cannot[u_idx, j] = ykk
            
            for j in range(m):
                # phi(j)=max(sum(sum(((S-Ai(:,:,j)).*W).^2)),eps);
                diff = (S - Ai[:, :, j]) * W
                val = np.sum(diff**2)
                phi[j] = max(val, np.finfo(float).eps)
                
            p = (1 / phi) / np.sum(1 / phi)
            
            fn1 = np.sum(ev[:c])
            fn2 = np.sum(ev[:c+1])
            
            if fn1 > 1e-9:
                lam = 2 * lam
            elif fn2 < 1e-11:
                lam = lam / 2
                F = F_old
            elif ii > 0: # ii>1 in MATLAB means at least 2 iterations done (indices 1, 2...)
                # Loop is 0-based ii=0,1... maxiter-1
                # if ii > 0 means we have done at least 2 passes
                break
                
        if qq > 0.5:
            qq = qq - delta
            
    return S, constraints
