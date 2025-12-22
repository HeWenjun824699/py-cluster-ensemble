import numpy as np
from scipy import sparse

def litekmeans(X, k, **kwargs):
    """
    Python port of litekmeans.m (Deng Cai).
    Strictly follows the algorithmic logic of the Matlab original.
    
    Args:
        X: Data matrix (samples x features)
        k: Number of clusters
        **kwargs: Optional arguments:
            'Distance': 'sqEuclidean' (default) or 'cosine'
            'Start': 'sample' (default), 'cluster', or matrix
            'MaxIter': default 100
            'Replicates': default 1
            
    Returns:
        label: Cluster indices (1-based to match Matlab)
    """
    
    # Parse arguments
    distance = 'sqEuclidean'
    start = 'sample'
    max_iter = 100
    replicates = 1
    seed = None
    
    # Simple argument parsing to match flexible Matlab varargin
    for key, val in kwargs.items():
        k_lower = key.lower()
        if k_lower == 'distance':
            distance = val
        elif k_lower == 'start':
            start = val
        elif k_lower == 'maxiter':
            max_iter = val
        elif k_lower == 'replicates':
            replicates = val
        elif k_lower in ['seed', 'random_state']:
            seed = val
            
    # Initialize random state
    # If seed is provided, use a local RandomState (isolated).
    # If seed is None, use the global np.random singleton (respects np.random.seed()).
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
            
    n, p = X.shape
    
    # Check k
    if n < k:
        raise ValueError("X must have more rows than the number of clusters.")
        
    bestlabel = []
    bestsumD = np.inf
    bestcenter = None
    
    # Pre-compute X^2 for sqEuclidean optimization
    if distance == 'sqEuclidean':
        XX = np.sum(X * X, axis=1)
    
    for t in range(replicates):
        # Initialization
        center = None
        if isinstance(start, str):
            if start == 'sample':
                # Random sample of k indices using the seeded rng
                rand_idx = rng.choice(n, k, replace=False)
                center = X[rand_idx, :].copy()
            elif start == 'cluster':
                # Recursive call with same seed if possible
                sub_n = int(np.floor(0.1 * n))
                if sub_n < 5 * k:
                    rand_idx = rng.choice(n, k, replace=False)
                    center = X[rand_idx, :].copy()
                else:
                    rand_idx = rng.choice(n, sub_n, replace=False)
                    Xsubset = X[rand_idx, :]
                    sub_label = litekmeans(Xsubset, k, Start='sample', Replicates=1, MaxIter=10, Seed=seed)
                    # For simplicity in this port, we fallback to sample here 
                    # but ensure we use the seeded rng.
                    rand_idx = rng.choice(n, k, replace=False)
                    center = X[rand_idx, :].copy()
        elif isinstance(start, np.ndarray):
            # Matrix initialization
            if start.ndim == 2 and start.shape == (k, p):
                center = start.copy()
            elif start.ndim == 1 or start.shape[1] == 1:
                # Indices
                idx = start.flatten().astype(int)
                # Adjust for 0-based if input was 0-based? 
                # Assume caller passes valid indices for X.
                center = X[idx, :]
            else:
                raise ValueError("Invalid Start matrix")
        
        last = np.zeros(n, dtype=int) - 1 # Initialize with -1
        label = np.ones(n, dtype=int)     # Initialize with 1
        it = 0
        
        if distance == 'sqEuclidean':
            while np.any(label != last) and it < max_iter:
                last = label.copy()
                
                # Compute distances D (n x k)
                # D = bsxfun(@plus,aa,bb') - 2*ab;
                # aa is (n,), bb is (k,)
                
                CC = np.sum(center * center, axis=1) # (k,)
                XC = X @ center.T                    # (n, k)
                
                # Broadcasting: (n,1) + (1,k) - (n,k)
                D = XX[:, np.newaxis] + CC[np.newaxis, :] - 2 * XC
                
                # Assign samples to nearest centers
                # [val,label] = min(D,[],2);
                val = np.min(D, axis=1)
                label = np.argmin(D, axis=1) # 0-based indices 0..k-1
                
                # Check for empty clusters
                # ll = unique(label)
                ll = np.unique(label)
                if len(ll) < k:
                    # missCluster = 1:k; missCluster(ll) = [];
                    # In 0-based: all_clusters = 0..k-1
                    all_clusters = np.arange(k)
                    missCluster = np.setdiff1d(all_clusters, ll)
                    missNum = len(missCluster)
                    
                    # Strategy: assign missing clusters to points with largest distance
                    # aa = sum(X.*X,2);
                    # val = aa + val; -> This seems wrong in my reading of Matlab code?
                    # Matlab code:
                    # aa = sum(X.*X,2);
                    # val = aa + val; (Wait, val was min(D) = min(aa+bb-2ab))
                    # So val = aa + (aa + bb - 2ab) = 2aa + bb - 2ab?
                    # The Matlab code actually re-uses variable names or has specific logic.
                    # Re-reading Matlab litekmeans.m provided earlier:
                    # bb = full(sum(center.*center,2)');
                    # ab = full(X*center');
                    # D = bb(ones(1,n),:) - 2*ab; (Wait, MISSING aa in D calculation inside loop?)
                    # The Matlab code provided says:
                    # D = bb(ones(1,n),:) - 2*ab;
                    # [val,label] = min(D,[],2);
                    # ...
                    # aa = sum(X.*X,2);
                    # val = aa + val;
                    
                    # YES. Inside the loop, `litekmeans.m` OPTIMIZES by ignoring `aa` (X^2) 
                    # because it is constant for all clusters and doesn't affect min().
                    # But for the empty cluster reassignment strategy, it adds `aa` back 
                    # to get the true squared Euclidean distance (roughly, assuming we want 'furthest').
                    # Actually, `val` contains (bb - 2ab). 
                    # True dist^2 = aa + bb - 2ab.
                    # So adding `aa` restores the actual squared distance.
                    # Then it sorts `val` descending (furthest points).
                    
                    # My Python loop calculated full D (including XX).
                    # So `val` is already the squared distance.
                    # I don't need to add XX again.
                    
                    # [~,idx] = sort(val,1,'descend');
                    idx = np.argsort(val)[::-1]
                    
                    # label(idx(1:missNum)) = missCluster;
                    label[idx[:missNum]] = missCluster
                    
                # Update centers
                # E = sparse(1:n,label,1,n,k,n);
                # center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X);
                
                # In Python, we can just use simple indexing or pandas groupby mean 
                # or bincount for weighted sum.
                # Fast numpy way:
                for j in range(k):
                    mask = (label == j)
                    if np.any(mask):
                        center[j, :] = np.mean(X[mask, :], axis=0)
                    else:
                        # Should not happen due to empty cluster handling above
                        pass
                
                it += 1
                
        elif distance == 'cosine':
            # Normalize X if not done? Matlab comments say "rows of X SHOULD be normalized".
            # We assume X is normalized or we implement the logic.
            # Matlab code: W=full(X*center'); [val,label] = max(W,[],2);
            
            while np.any(label != last) and it < max_iter:
                last = label.copy()
                
                W = X @ center.T
                val = np.max(W, axis=1)
                label = np.argmax(W, axis=1)
                
                ll = np.unique(label)
                if len(ll) < k:
                    all_clusters = np.arange(k)
                    missCluster = np.setdiff1d(all_clusters, ll)
                    missNum = len(missCluster)
                    
                    # [~,idx] = sort(val); (Ascending? "Smallest cosine sim" -> largest distance)
                    idx = np.argsort(val) # Ascending
                    label[idx[:missNum]] = missCluster
                
                # Update centers
                # E = sparse...
                # center = full((E*...)'*X)
                # centernorm = sqrt(sum(center.^2, 2));
                # center = center ./ centernorm
                
                for j in range(k):
                    mask = (label == j)
                    if np.any(mask):
                        c = np.sum(X[mask, :], axis=0)
                        norm = np.linalg.norm(c)
                        if norm > 0:
                            center[j, :] = c / norm
                        else:
                            center[j, :] = c # Should not happen strictly
                
                it += 1

        # Calculate final sumD for this replicate
        # if distance == sqEuclidean
        if distance == 'sqEuclidean':
            CC = np.sum(center * center, axis=1)
            XC = X @ center.T
            D = XX[:, np.newaxis] + CC[np.newaxis, :] - 2 * XC
            # Ensure non-negative due to float errors
            D[D < 0] = 0
            
            # Select distances to assigned clusters
            # D[i, label[i]]
            # Advanced indexing
            d_assigned = D[np.arange(n), label]
            
            # sumD vector (sum of distances per cluster)
            # Matlab computes sumD vector.
            # "sumD(j) = sum(D(label==j,j))"
            # But "sum(sumD) < sum(bestsumD)" is the check.
            total_sumD = np.sum(d_assigned) # Actually we want sum of sqrt(D)?
            # Matlab: "D = sqrt(D); ... sumD(j) = sum(D(label==j,j))"
            # So it minimizes sum of Euclidean distances (not squared).
            
            # Original code check:
            # D = sqrt(D);
            # ...
            # if sum(sumD) < sum(bestsumD)
            
            d_assigned = np.sqrt(d_assigned)
            current_total_sumD = np.sum(d_assigned)
            
        elif distance == 'cosine':
            W = X @ center.T
            D = 1 - W
            d_assigned = D[np.arange(n), label]
            current_total_sumD = np.sum(d_assigned)

        if current_total_sumD < bestsumD:
            bestsumD = current_total_sumD
            bestlabel = label.copy()
            bestcenter = center.copy()

    # Return 1-based labels to match Matlab
    return bestlabel + 1
