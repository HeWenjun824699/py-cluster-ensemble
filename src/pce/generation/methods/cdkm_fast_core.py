import numpy as np


def cdkm_fast_core(X, label, c=None):
    """
    Coordinate Descent Method for k-means (Fast Version).

    Ref: F. Nie, et al. "Coordinate descent method for k-means", TPAMI 2021.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix.
    label : array-like, shape (n_samples,)
        Initial cluster labels (0-based indices).
    c : int, optional
        Number of clusters. If None, inferred from label.

    Returns
    -------
    Y_label : ndarray
        Refined cluster labels (0-based).
    iter_num : int
        Number of iterations performed.
    obj_max : ndarray
        Objective function value history.
    """
    X = np.array(X, dtype=np.float64)
    n = X.shape[0]

    # Ensure label is 0-based integer array
    label = np.array(label, dtype=int).flatten()

    if c is None:
        c = len(np.unique(label))

    # --- Initialization ---

    # 1. Transform label into indicator matrix Y (n x c)
    # Using dense matrix for speed in coordinate descent updates
    Y = np.zeros((n, c), dtype=np.float64)
    Y[np.arange(n), label] = 1.0

    # 2. Cluster counts: aa (1 x c) -> in Python (c,)
    aa = np.sum(Y, axis=0)

    # 3. Similarity Matrix BBB (n x n)
    # MATLAB: BBB = 2 * (X' * X) where X is d*n
    # Python: BBB = 2 * (X @ X.T) where X is n*d
    # Note: This is O(n^2 d) and O(n^2) memory.
    BBB = 2 * (X @ X.T)

    # 4. Diagonal terms
    # XX = diag(BBB) ./ 2
    XX = np.diag(BBB) / 2.0

    # 5. Projection of similarity onto clusters
    # BBUU = BBB * Y  (n x c)
    BBUU = BBB @ Y

    # 6. Objective term
    # ybby = diag(Y' * BBUU / 2) -> sum(Y * BBUU, axis=0) / 2
    ybby = np.sum(Y * BBUU, axis=0) / 2.0

    # 7. Initial Objective Value
    obj_max = []
    # MATLAB: sum(ybby ./ aa')
    # Handle division by zero if any cluster is empty (though unlikely in init)
    with np.errstate(divide='ignore', invalid='ignore'):
        current_obj = np.sum(np.divide(ybby, aa, where=aa != 0))
    obj_max.append(current_obj)

    # --- Coordinate Descent Loop ---

    last_label = np.zeros(n) - 1  # Init with -1 so first check fails
    iter_num = 0

    # Loop until convergence (labels don't change)
    while np.any(label != last_label):
        last_label = label.copy()

        for i in range(n):
            m = label[i]  # Current cluster index

            # If singleton cluster, cannot move
            if aa[m] <= 1:  # Floating point safety, though aa is integer-like
                continue

            # Calculate objective changes
            # y_i is the i-th row of Y (one-hot, 1 at m, 0 elsewhere)
            y_i = Y[i, :]
            mask = 1.0 - y_i

            # MATLAB: V21 = ybby' + (BBUU(i,:) + XX(i)) .* (1-Y(i,:));
            V21 = ybby + (BBUU[i, :] + XX[i]) * mask

            # MATLAB: V11 = ybby' - (BBUU(i,:) - XX(i)) .* Y(i,:);
            # Note: Y(i,:) is 1 only at m. So V11 is ybby everywhere except at m.
            V11 = ybby - (BBUU[i, :] - XX[i]) * y_i

            # MATLAB: delta = V21./(aa+1-Y(i,:)) - V11./(aa-Y(i,:));
            # term1: (aa + 1) for others, (aa) for self (but masked out in V21)
            term1 = np.divide(V21, aa + mask, where=(aa + mask) != 0)

            # term2: (aa) for others, (aa - 1) for self
            term2 = np.divide(V11, aa - y_i, where=(aa - y_i) != 0)

            delta = term1 - term2

            # Find best cluster to move to
            q = np.argmax(delta)

            if m != q:
                # Update State

                # Update counts
                aa[q] += 1
                aa[m] -= 1

                # Update ybby terms
                ybby[m] = V11[m]
                ybby[q] = V21[q]

                # Update Indicator Y
                Y[i, m] = 0.0
                Y[i, q] = 1.0
                label[i] = q

                # Update BBUU (Smart Update)
                # Avoid recomputing full matrix multiplication
                # BBUU(:, m) -= BBB(:, i)
                # BBUU(:, q) += BBB(:, i)
                # BBB[:, i] is the i-th column (same as row for symmetric matrix)
                col_i = BBB[:, i]
                BBUU[:, m] -= col_i
                BBUU[:, q] += col_i

        iter_num += 1

        # Compute objective
        with np.errstate(divide='ignore', invalid='ignore'):
            current_obj = np.sum(np.divide(ybby, aa, where=aa != 0))
        obj_max.append(current_obj)

    return label, iter_num, np.array(obj_max)
