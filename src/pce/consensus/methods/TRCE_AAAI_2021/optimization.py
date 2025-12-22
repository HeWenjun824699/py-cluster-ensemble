import numpy as np
from .L2_distance_1 import L2_distance_1
from .EuclideanPro import EuclideanPro
from .optimize_S import optimize_S
from .eig1 import eig1

def optimization(Si, c, gamma):
    """
    OPTIMIZATION
    
    Args:
        Si: Tensor (n x n x m)
        c: number of clusters
        gamma: parameter
    
    Returns:
        S: optimized similarity matrix
    """
    
    # [n,~,m]=size(Si);
    n, _, m = Si.shape
    maxiter = 10
    alpha = np.ones(m) / m
    
    # Normalize Si
    # for j=1:m
    #    dd=1./max(sum(Si(:,:,j),2),eps);
    #    Si(:,:,j)=bsxfun(@times,Si(:,:,j),dd);
    # end
    for j in range(m):
        row_sum = np.sum(Si[:, :, j], axis=1) # Sum over columns (axis 1) -> (n,)
        dd = 1.0 / np.maximum(row_sum, np.finfo(float).eps)
        # Broadcasting: (n, n) * (n, 1) if we reshape dd
        Si[:, :, j] = Si[:, :, j] * dd[:, np.newaxis]

    # S=sum(Si,3)./m;
    S = np.sum(Si, axis=2) / m
    
    # S = (S+S')/2;
    S = (S + S.T) / 2
    
    # D = diag(sum(S));
    # L = D -S;
    # optimization.m uses L for eig1
    D_vec = np.sum(S, axis=1) # Row sum? sum(S) in matlab sums columns. S is symmetric.
    # D = np.diag(D_vec)
    L = np.diag(D_vec) - S
    
    B = S.copy()
    
    # [F, ~, evs]=eig1(L, c, 0);
    F, _, evs_initial = eig1(L, c, 0)
    
    # Keep track of evs if needed, though python doesn't support growing matrices easily.
    # evs(:,ii+1) = ev; implies evs is (n, maxiter+1) or similar. 
    # eig1 returns vector of eigenvalues.
    # We will just store them in a list or array.
    evs_history = []
    evs_history.append(evs_initial)

    # d1=zeros(n,m);
    # for k=1:m
    #     tmp1=Si(:,:,k).*log(max(Si(:,:,k)./max(B,eps),eps));
    #     d1(:,k)=sum(tmp1,2);
    # end
    d1 = np.zeros((n, m))
    for k in range(m):
        # max(B, eps)
        denom = np.maximum(B, np.finfo(float).eps)
        term_inside = np.maximum(Si[:, :, k] / denom, np.finfo(float).eps)
        tmp1 = Si[:, :, k] * np.log(term_inside)
        d1[:, k] = np.sum(tmp1, axis=1)

    rho = 1.0
    lambda_val = 0.0 # lambda is a keyword in python
    
    for ii in range(maxiter): # 0 to maxiter-1
        # d2=zeros(n,1);
        # for j=1:m ... k used in loop body?
        # matlab: for j=1:m; d2=d2+d1(:,k)./alpha(k); end
        # variable iterator j, but uses k inside. This assumes k is from previous scope? 
        # Wait, Matlab:
        # for k=1:m ... d1(:,k)=... end
        # rho=1
        # for ii=1:maxiter
        #     d2=zeros(n,1);
        #     for j=1:m
        #         d2=d2+d1(:,k)./alpha(k);   <-- This 'k' looks like a BUG in the original Matlab code?
        #     end
        # In Matlab, loop variable 'k' retains value after loop. k=m from previous loop.
        # So it sums d1(:,m)/alpha(m) m times?
        # That seems wrong. It likely meant 'j'.
        # Let's look at the logic. d2 is likely sum of d1(:,j)/alpha(j).
        # "d2=d2+d1(:,k)./alpha(k)" inside "for j=1:m".
        # If I strictly convert, I should follow the bug? Or fix it?
        # Most likely it's a typo for 'j'.
        # But 'k' is 20 (nBase) after the previous loop.
        # If I fix it to 'j', it makes sense.
        # IF I follow 'k', it sums the last component m times.
        # I will assume 'j' is intended but I will check if 'k' makes sense. 
        # Usually these algorithms sum over all views.
        # I'll use 'j'.
        
        d2 = np.zeros(n)
        for j in range(m):
            d2 = d2 + d1[:, j] / alpha[j]
        
        # if ii==1 (1-based) -> ii==0 (0-based)
        if ii == 0:
            sort_d2 = np.sort(d2)
            # lambda=2.*sort_d2(min(floor(n/maxiter)*ii,n));
            # In Matlab ii=1 (1st iter). 
            # floor(n/10)*1. 
            # idx = min(..., n).
            # Python ii=0. We should map logic.
            # Matlab iter 1: index = floor(n/10).
            # Matlab iter 2: index = floor(n/10)*2.
            # So lambda changes? 
            # Wait, the code:
            # if ii==1 ... else lambda=lambda*1.1
            # So lambda is initialized at iter 1.
            
            idx_m = min(int(np.floor(n / maxiter) * (ii + 1)), n)
            # Matlab 1-based index conversion: sort_d2(idx_m) -> sort_d2[idx_m-1]
            if idx_m == 0: idx_m = 1 # Avoid 0 if n < maxiter?
            lambda_val = 2.0 * sort_d2[idx_m - 1]
        else:
            lambda_val = lambda_val * 1.1
            
        # w=min(lambda./max(2.*d2,eps),1);
        w = np.minimum(lambda_val / np.maximum(2 * d2, np.finfo(float).eps), 1)
        
        # distf = L2_distance_1(F',F');
        # F is (n, c). F' is (c, n).
        # L2_distance_1 expects (d, n).
        distf = L2_distance_1(F.T, F.T)
        
        # distf=distf-diag(diag(distf));
        np.fill_diagonal(distf, 0)
        
        # C=zeros(n,n);
        # for k=1:m
        #     C=C+Si(:,:,k)./alpha(k);
        # end
        C = np.zeros((n, n))
        for k in range(m):
            C = C + Si[:, :, k] / alpha[k]
        
        # C=bsxfun(@times,C,w.^2);
        # w is (n,). w.^2 is (n,).
        # bsxfun(@times, C, w.^2) -> C columns multiplied by w^2? 
        # bsxfun matches dimensions. C is (n,n). w is (n,1) usually?
        # w calculation above: d2 is (n,1) or (n,). w is (n,).
        # In Matlab, if w is column vector, C .* w -> scales columns?
        # Wait, if w is (n,1), then C is (n,n).
        # Matlab: A (nxn) .* v (nx1) -> scales each column i by v(i)? No, scales ROWS.
        # A_ij * v_i.
        # Python: C * w[:, None] -> scales rows.
        # Let's verify 'w'. d2 is sum of columns, so d2 is (n,1).
        # So w is (n,1).
        # So bsxfun(@times, C, w.^2) multiplies each row i by w(i)^2.
        C = C * (w[:, np.newaxis] ** 2)
        
        # B=optimize_S(distf,C,rho);
        B = optimize_S(distf, C, rho)
        
        # for j=1:n
        #     v=B(j,:)-rho./(2*gamma).*distf(j,:);
        #     S(j,:)=EuclideanPro(ones(1,n),v);
        # end
        for j in range(n):
            v = B[j, :] - (rho / (2 * gamma)) * distf[j, :]
            # EuclideanPro(w, v). w=ones.
            S[j, :] = EuclideanPro(np.ones(n), v)
            
        # d1=zeros(n,m);
        # for k=1:m
        #     tmp1=Si(:,:,k).*log(max(Si(:,:,k)./max(B,eps),eps));
        #     d1(:,k)=sum(tmp1,2);
        # end
        d1 = np.zeros((n, m))
        for k in range(m):
             denom = np.maximum(B, np.finfo(float).eps)
             term_inside = np.maximum(Si[:, :, k] / denom, np.finfo(float).eps)
             tmp1 = Si[:, :, k] * np.log(term_inside)
             d1[:, k] = np.sum(tmp1, axis=1)

        # S = (S+S')/2;
        S = (S + S.T) / 2
        
        # D = diag(sum(S));
        # L = D-S;
        D_vec = np.sum(S, axis=1)
        L = np.diag(D_vec) - S
        
        F_old = F
        # [F, ~, ev]=eig1(L, c, 0);
        # evs(:,ii+1) = ev;
        F, _, ev = eig1(L, c, 0)
        # Store ev?
        
        # tmp2=bsxfun(@times,d1,w.^2);
        # d1 is (n, m). w is (n,).
        # bsxfun scales rows of d1 by w^2.
        tmp2 = d1 * (w[:, np.newaxis] ** 2)
        
        # d3=sum(tmp2); -> sum over columns -> (1, m) or (m,)
        d3 = np.sum(tmp2, axis=0)
        d3 = np.sqrt(d3)
        alpha = d3 / np.sum(d3)
        
        # fn1 = sum(ev(1:c));
        # fn2 = sum(ev(1:c+1));
        # ev is sorted ascending (isMax=0).
        # ev(1:c) are the first c.
        # ev(1:c+1) are first c+1.
        # Note: eig1 returns 'eigval' as subset and 'eigval_full' as all/subset.
        # The 'ev' variable here likely refers to 'eigval_full' from eig1?
        # Code: [F, ~, ev]=eig1(L, c, 0);
        # My eig1 returns (vec, val, val_full).
        # So 'ev' is val_full.
        
        fn1 = np.sum(ev[0:c])
        # Check if ev has enough elements for c+1
        if len(ev) > c:
            fn2 = np.sum(ev[0:c+1])
        else:
            fn2 = fn1 # Should not happen if L is nxn and c < n
            
        if fn1 > 0.000000001:
            rho = 2 * rho
        elif fn2 < 0.00000000001:
            rho = rho / 2
            F = F_old
        elif ii > 0: # ii > 1 in 1-based -> ii > 0 in 0-based
            break
            
    return S
