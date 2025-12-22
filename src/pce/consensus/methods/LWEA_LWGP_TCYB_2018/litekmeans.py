import numpy as np
from scipy import sparse


def litekmeans(X, k, distance='sqeuclidean', start='sample', maxiter=100, replicates=1, clustermaxiter=10):
    """
    LITEKMEANS K-means clustering, accelerated by numpy matrix operations.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix.
    k : int
        The number of clusters.
    distance : {'sqeuclidean', 'cosine'}, optional
        Distance measure. Default is 'sqeuclidean'.
    start : str or array-like, optional
        Method for initialization:
        - 'sample': Select K observations from X at random (default).
        - 'cluster': Perform preliminary clustering on random 10% subsample.
        - matrix/array: K-by-P matrix of starting locations or K indices.
    maxiter : int, optional
        Maximum number of iterations. Default is 100.
    replicates : int, optional
        Number of times to repeat clustering. Default is 1.
    clustermaxiter : int, optional
        Max iterations for preliminary phase if start='cluster'. Default is 10.

    Returns
    -------
    label : ndarray, shape (n_samples,)
        Cluster indices of each point (0 to k-1).
    center : ndarray, shape (k, n_features)
        Cluster centroid locations.
    bCon : bool
        Whether the iteration converged.
    sumD : ndarray, shape (k,)
        Within-cluster sums of point-to-centroid distances.
    D : ndarray, shape (n_samples, k)
        Distances from each point to every centroid.
    """

    X = np.array(X, dtype=np.float64)
    n, p = X.shape

    # --- Parameter Validation ---
    if isinstance(distance, str):
        distance = distance.lower()
        if distance not in ['sqeuclidean', 'cosine']:
            raise ValueError(f"Unknown 'Distance' parameter value: {distance}")
    else:
        raise ValueError("The 'Distance' parameter value must be a string.")

    init_center = None
    if isinstance(start, str):
        start = start.lower()
        if start not in ['sample', 'cluster']:
            raise ValueError(f"Unknown 'Start' parameter value: {start}")
        if start == 'cluster':
            if np.floor(0.1 * n) < 5 * k:
                start = 'sample'
    elif isinstance(start, (np.ndarray, list)):
        start = np.array(start)
        if start.shape[1] == p:
            init_center = start
        elif start.ndim == 1 or start.shape[1] == 1:
            init_center = X[start.flatten(), :]
        else:
            raise ValueError("The 'Start' matrix must have the same number of columns as X.")

        if k is None:
            k = init_center.shape[0]
        elif k != init_center.shape[0]:
            raise ValueError("The 'Start' matrix must have K rows.")
        start = 'numeric'
        replicates = 1
    else:
        raise ValueError("The 'Start' parameter value must be a string or a numeric matrix.")

    if not np.isscalar(k) or k <= 0 or k % 1 != 0:
        # Note: In Python, we ensure k is integer-like
        if not (isinstance(k, int) or k.is_integer()):
            raise ValueError("K must be a positive integer.")
    k = int(k)

    if n < k:
        raise ValueError("X must have more rows than the number of clusters.")

    # --- Main Loop ---
    bestlabel = None
    bestcenter = None
    bestsumD = None
    bestD = None
    bCon = False

    for t in range(replicates):
        # Initialization
        if start == 'sample':
            # Equivalent to randsample(n, k)
            indices = np.random.choice(n, k, replace=False)
            center = X[indices, :]
        elif start == 'cluster':
            subset_size = int(np.floor(0.1 * n))
            indices = np.random.choice(n, subset_size, replace=False)
            Xsubset = X[indices, :]
            # Recursive call
            # Note: MATLAB [dump, center] means we only need the center
            _, center, _, _, _ = litekmeans(Xsubset, k, start='sample',
                                            replicates=1, maxiter=clustermaxiter,
                                            distance=distance)
        elif start == 'numeric':
            center = init_center.copy()

        last = np.zeros(n) - 1  # Initialize with -1 so first comparison fails
        label = np.zeros(n, dtype=np.float64)
        it = 0

        # Optimization
        if distance == 'sqeuclidean':
            while np.any(label != last) and it < maxiter:
                last = label.copy()

                # bb = sum(center.*center,2)'
                bb = np.sum(center ** 2, axis=1)
                # ab = X*center'
                ab = X @ center.T

                # D = bb(ones(1,n),:) - 2*ab
                # Broadcasting: (k,) becomes (1,k) implicit matching (n,k)
                D = bb[None, :] - 2 * ab

                # [val,label] = min(D,[],2)
                label = np.argmin(D, axis=1)
                val = np.min(D, axis=1)

                # Handle empty clusters
                ll = np.unique(label)
                if len(ll) < k:
                    # missCluster = 1:k; missCluster(ll) = [];
                    # In Python 0-based: set(0..k-1) - set(ll)
                    missCluster = list(set(range(k)) - set(ll))
                    missNum = len(missCluster)

                    # aa = sum(X.*X,2)
                    aa = np.sum(X ** 2, axis=1)
                    val = aa + val  # Actual squared distance

                    # [dump,idx] = sort(val,1,'descend')
                    idx = np.argsort(val)[::-1]
                    label[idx[:missNum]] = missCluster

                # Update centers using sparse matrix logic
                # E = sparse(1:n,label,1,n,k,n)
                # Python: row=range(n), col=label, data=1
                E = sparse.csr_matrix((np.ones(n), (np.arange(n), label)), shape=(n, k))

                # center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X)
                # sum(E,1) in MATLAB sums columns (1st dim). Here E is n x k.
                # We need number of points in each cluster.
                count = np.array(E.sum(axis=0)).flatten()

                # Avoid division by zero
                count = np.where(count == 0, 1, count)

                # (E * diag(1/count)).T * X  => (diag(1/count) * E.T) * X
                # Equivalently: (E.T @ X) / count[:, None]
                center = (E.T @ X) / count[:, None]

                it += 1

            if it < maxiter:
                bCon = True

            # Final calculation for this replicate
            aa = np.sum(X ** 2, axis=1)
            if replicates > 1 or bestlabel is None:
                if it >= maxiter:
                    bb = np.sum(center ** 2, axis=1)
                    ab = X @ center.T
                    D = aa[:, None] + bb[None, :] - 2 * ab
                    D[D < 0] = 0
                else:
                    # In converged case, D (bb-2ab) is already computed, just add aa
                    # Note: D variable in loop didn't have aa.
                    # Recomputing to be safe and match logic structure:
                    bb = np.sum(center ** 2, axis=1)
                    ab = X @ center.T
                    D = aa[:, None] + bb[None, :] - 2 * ab
                    D[D < 0] = 0

                D = np.sqrt(D)
                sumD = np.zeros(k)
                for j in range(k):
                    sumD[j] = np.sum(D[label == j, j])

                if bestlabel is None or np.sum(sumD) < np.sum(bestsumD):
                    bestlabel = label.copy()
                    bestcenter = center.copy()
                    bestsumD = sumD.copy()
                    bestD = D.copy()

        elif distance == 'cosine':
            while np.any(label != last) and it < maxiter:
                last = label.copy()

                # W=full(X*center')
                W = X @ center.T

                # [val,label] = max(W,[],2)
                label = np.argmax(W, axis=1)
                val = np.max(W, axis=1)

                ll = np.unique(label)
                if len(ll) < k:
                    missCluster = list(set(range(k)) - set(ll))
                    missNum = len(missCluster)

                    # [dump,idx] = sort(val) (ascending for cosine similarity mean worst match)
                    idx = np.argsort(val)
                    label[idx[:missNum]] = missCluster

                E = sparse.csr_matrix((np.ones(n), (np.arange(n), label)), shape=(n, k))

                # center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X)
                # Compute raw sum first
                center = E.T @ X

                # Normalize centers
                # centernorm = sqrt(sum(center.^2, 2))
                centernorm = np.sqrt(np.sum(center ** 2, axis=1))
                centernorm = np.where(centernorm == 0, 1, centernorm)
                center = center / centernorm[:, None]

                it += 1

            if it < maxiter:
                bCon = True

            if replicates > 1 or bestlabel is None:
                W = X @ center.T
                D = 1 - W
                sumD = np.zeros(k)
                for j in range(k):
                    sumD[j] = np.sum(D[label == j, j])

                if bestlabel is None or np.sum(sumD) < np.sum(bestsumD):
                    bestlabel = label.copy()
                    bestcenter = center.copy()
                    bestsumD = sumD.copy()
                    bestD = D.copy()

    return bestlabel, bestcenter, bCon, bestsumD, bestD
