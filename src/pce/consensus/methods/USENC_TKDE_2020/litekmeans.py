import numpy as np
from scipy import sparse

def litekmeans(X, k, distance='sqeuclidean', start='sample', max_iter=100, replicates=1, random_state=None):
    """
    LITEKMEANS K-means clustering, accelerated by numpy matrix operations.
    
    Parameters:
    -----------
    X : numpy.ndarray or scipy.sparse matrix
        n_samples x n_features data matrix.
    k : int
        Number of clusters.
    distance : str, optional (default='sqeuclidean')
        Distance measure: 'sqeuclidean' or 'cosine'.
    start : str or numpy.ndarray, optional (default='sample')
        Method to choose initial centers: 'sample', 'cluster', or a matrix of centers.
    max_iter : int, optional (default=100)
        Maximum number of iterations.
    replicates : int, optional (default=1)
        Number of times to repeat clustering with new initial centroids.
    random_state : int, RandomState instance or None, optional
        Determines random number generation for centroid initialization.
        
    Returns:
    --------
    label : numpy.ndarray
        Cluster indices (0-based) for each point.
    center : numpy.ndarray
        Cluster centroids.
    sumD : numpy.ndarray
        Within-cluster sum of distances.
    D : numpy.ndarray
        Distances from each point to every centroid.
    """
    
    # Ensure X is valid
    X = np.atleast_2d(X)
    n, p = X.shape
    
    if random_state is None:
        rng = np.random
    elif hasattr(random_state, 'randint'):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    if n < k:
        raise ValueError("X must have more rows than the number of clusters.")

    best_label = None
    best_center = None
    best_sumD = None
    best_D = None
    
    # Loop for replicates
    for t in range(replicates):
        # Initialization
        center = None
        if isinstance(start, str):
            if start == 'sample':
                center_indices = rng.choice(n, k, replace=False)
                center = X[center_indices, :]
            elif start == 'cluster':
                # Recursive initialization on a subset
                subset_size = min(n, int(max(0.1 * n, 5 * k))) # Heuristic from original code
                subset_indices = rng.choice(n, subset_size, replace=False)
                Xsubset = X[subset_indices, :]
                _, center, _, _ = litekmeans(Xsubset, k, distance=distance, start='sample', replicates=1, random_state=rng)
            elif start == 'numeric': # Should be caught by isinstance check below usually
                pass 
        elif isinstance(start, np.ndarray):
            if start.shape[1] != p:
                 raise ValueError("The 'Start' matrix must have the same number of columns as X.")
            if start.shape[0] == k:
                center = start
            elif start.shape[0] == 1 or start.shape[1] == 1:
                # Indices provided
                center = X[start.flatten(), :]
            else:
                 raise ValueError("The 'Start' matrix must have K rows.")
        
        if center is None:
             raise ValueError("Invalid 'start' parameter.")

        last_label = np.zeros(n, dtype=int) - 1
        label = np.zeros(n, dtype=int)
        it = 0
        bCon = False
        
        # Main Loop
        if distance == 'sqeuclidean':
            while np.any(label != last_label) and it < max_iter:
                last_label = label.copy()
                
                # Compute distances: ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x*c'
                # Note: X is (n, p), center is (k, p)
                # bb: (1, k) sum of squares of centers
                bb = np.sum(center**2, axis=1).reshape(1, k)
                # ab: (n, k) product X * center.T
                ab = X @ center.T
                
                # D: (n, k) squared euclidean distances
                # We add bb to every row of -2*ab using broadcasting.
                # Then we add aa (sum of squares of X) later or just ignore it for assignment 
                # because it's constant for all clusters for a given point.
                # Original code: D = bb(ones(1,n),:) - 2*ab;
                D = bb - 2 * ab
                
                # Assign samples to nearest centers
                # min over axis 1 (columns/clusters)
                label = np.argmin(D, axis=1)
                val = np.min(D, axis=1) # Minimal 'distance' (minus x^2 term)
                
                # Handle empty clusters
                ll = np.unique(label)
                if len(ll) < k:
                    # Original logic: identify missing clusters
                    missCluster = list(set(range(k)) - set(ll))
                    missNum = len(missCluster)
                    
                    # Original logic: split the points with largest current distance contribution
                    # In original: aa = sum(X.*X,2); val = aa + val; 
                    # val was (bb - 2ab), so val + aa is the actual squared distance.
                    aa = np.sum(X**2, axis=1)
                    actual_dist = aa + val
                    
                    # Sort descending
                    idx = np.argsort(actual_dist)[::-1]
                    label[idx[:missNum]] = missCluster
                
                # Update centers
                # E = sparse(1:n,label,1,n,k,n)
                # center = (E*spdiags(1./sum(E,1)',0,k,k))'*X
                
                # In Python, we can do this efficiently:
                # E is n x k indicator matrix. 
                data = np.ones(n)
                row_ind = np.arange(n)
                col_ind = label
                E = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, k))
                
                # Cluster counts
                counts = np.array(E.sum(axis=0)).flatten()
                # Avoid divide by zero (though empty clusters handled above, safe check)
                counts[counts == 0] = 1 
                
                # E.T @ X sums the points for each cluster
                # Then divide by counts
                center = (E.T @ X) / counts[:, None]
                
                it += 1
            
            if it < max_iter:
                bCon = True
            
            # Final distance calculation for this replicate
            aa = np.sum(X**2, axis=1).reshape(n, 1)
            bb = np.sum(center**2, axis=1).reshape(1, k)
            ab = X @ center.T
            D = aa + bb - 2 * ab
            D[D < 0] = 0
            D = np.sqrt(D)
            
            sumD_current = np.zeros(k)
            for j in range(k):
                sumD_current[j] = np.sum(D[label == j, j])
                
        elif distance == 'cosine':
             while np.any(label != last_label) and it < max_iter:
                last_label = label.copy()
                
                # W = X * center'
                W = X @ center.T
                
                # Maximize cosine similarity -> min (1 - cosine)
                # assign samples to nearest centers (max W)
                label = np.argmax(W, axis=1)
                val = np.max(W, axis=1)
                
                # Handle empty clusters
                ll = np.unique(label)
                if len(ll) < k:
                    missCluster = list(set(range(k)) - set(ll))
                    missNum = len(missCluster)
                    
                    # Sort ascending (because val is similarity, low similarity = "far")
                    idx = np.argsort(val) # Smallest similarity first
                    label[idx[:missNum]] = missCluster
                
                # Update centers
                data = np.ones(n)
                row_ind = np.arange(n)
                col_ind = label
                E = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, k))
                
                center = E.T @ X
                # Normalize centers to unit norm
                centernorm = np.sqrt(np.sum(center**2, axis=1)).reshape(k, 1)
                centernorm[centernorm == 0] = 1 # Avoid div by zero
                center = center / centernorm
                
                it += 1
            
             if it < max_iter:
                bCon = True
             
             W = X @ center.T
             D = 1 - W
             sumD_current = np.zeros(k)
             for j in range(k):
                 sumD_current[j] = np.sum(D[label == j, j])

        # Keep best replicate
        if best_label is None or np.sum(sumD_current) < np.sum(best_sumD):
            best_label = label
            best_center = center
            best_sumD = sumD_current
            best_D = D
            
    return best_label, best_center, best_sumD, best_D
