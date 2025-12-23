import numpy as np
from scipy.sparse import issparse, csc_matrix, lil_matrix, coo_matrix

from .CDKM_TPAMI_2022.compute_Hc import compute_Hc
from .CDKM_TPAMI_2022.Y_Initialize import Y_Initialize

def cdkm_core(X, label, c):
    """
    Coordinate Descent Method for k-means (Fast Version)

    Parameters:
    X (numpy.ndarray or scipy.sparse.spmatrix): d x n data matrix.
    label (numpy.ndarray): n x 1 initial labels.
    c (int): Number of clusters.

    Returns:
    tuple: (Y_label, iter_num, obj_max)
        Y_label: Final label vector (n,).
        iter_num: Number of iterations.
        obj_max: List of objective function values.
    """
    # X is d x n
    if issparse(X):
        d, n = X.shape
    else:
        d, n = X.shape

    # Ensure label is flat and 0-based for Python indexing
    label = np.array(label).flatten().astype(int)
    if label.min() == 1:
        label = label - 1

    # Initialize Y as a sparse indicator matrix
    # rows = 0..n-1, cols = label, data = 1
    # Using LIL format as we might update it, though logic below avoids heavy sparse updates
    rows = np.arange(n)
    cols = label
    data = np.ones(n)
    # Fix: lil_matrix doesn't support (data, (r, c)) constructor directly.
    # Use coo_matrix first then convert.
    Y = coo_matrix((data, (rows, cols)), shape=(n, c)).tolil()

    last = np.zeros_like(label) - 1
    iter_num = 0
    obj_max = []

    # aa = sum(Y, 1) -> Sum of columns (cluster sizes)
    aa = np.array(Y.sum(axis=0)).flatten()

    # [~, label] = max(Y, [], 2) -> Already have label, but strictly following logic:
    # In python argmax on sparse matrix can be tricky, but we have label input.
    # The MATLAB code re-derives label from Y.

    # BBB = 2 * (X' * X) -> n x n kernel matrix
    # If X is sparse d x n, X.T is n x d. Result is n x n.
    # Warning: If n is large, this is memory intensive (n^2).
    if issparse(X):
        BBB = 2 * (X.T @ X)
    else:
        BBB = 2 * np.dot(X.T, X)

    # XX = diag(BBB) ./ 2
    if issparse(BBB):
        XX = BBB.diagonal() / 2.0
        # Convert BBB to CSC or CSR for efficient slicing/multiplication
        BBB = BBB.tocsc()
    else:
        XX = np.diag(BBB) / 2.0

    # BBUU = BBB * Y
    BBUU = BBB @ Y
    if issparse(BBUU):
        BBUU = BBUU.toarray()

    # ybby = diag(Y' * BBUU / 2)
    # Using element-wise multiplication sum which is faster for diag(A @ B) -> sum(A.T * B, axis=...)
    # Here diag(Y.T @ BBUU). Y is n x c, BBUU is n x c.
    # Diagonal element k is dot product of k-th col of Y and k-th col of BBUU.
    # Since Y is indicator, k-th col of Y has 1s at indices where label==k.
    # So ybby[k] = sum(BBUU[i, k]) for all i where label[i] == k.
    # But let's stick to matrix op for consistency if fast enough.
    # (Y.T @ BBUU).diagonal()
    ybby = (Y.T @ BBUU).diagonal() / 2.0

    # Initial objective
    # obj_max(1) = sum(ybby ./ aa')
    current_obj = np.sum(ybby / aa)
    obj_max.append(current_obj)

    while np.any(label != last):
        last = label.copy()
        for i in range(n):
            m = label[i]
            if aa[m] == 1:
                continue

            # MATLAB:
            # V21 = ybby' + (BBUU(i,:) + XX(i)) .* (1 - Y(i,:));
            # V11 = ybby' - (BBUU(i,:) - XX(i)) .* Y(i,:);
            # delta = V21./(aa+1-Y(i,:)) - V11./(aa-Y(i,:));

            # Python Implementation:
            # BBUU[i, :] is row vector (c,)
            # XX[i] is scalar
            # Y[i, :] is one-hot at m

            # Pre-calculate terms
            BBUU_i = BBUU[i] # Dense (c,)
            XX_i = XX[i]

            # V21: Cost if we add i to cluster k
            # If k != m: (1 - Y(i,k)) is 1. Term is ybby[k] + BBUU_i[k] + XX_i
            # If k == m: (1 - Y(i,m)) is 0. Term is ybby[m]
            V21 = ybby + BBUU_i + XX_i
            V21[m] = ybby[m]

            # V11: Cost if we remove i from cluster k
            # If k != m: Y(i,k) is 0. Term is ybby[k]
            # If k == m: Y(i,m) is 1. Term is ybby[m] - (BBUU_i[m] - XX_i)
            V11 = ybby.copy()
            V11[m] = ybby[m] - (BBUU_i[m] - XX_i)

            # Denominators for delta
            # aa + 1 - Y(i,:)
            # If k != m: aa[k] + 1
            # If k == m: aa[m]
            denom1 = aa + 1.0
            denom1[m] = aa[m]

            # aa - Y(i,:)
            # If k != m: aa[k]
            # If k == m: aa[m] - 1
            denom2 = aa.copy().astype(float)
            denom2[m] = aa[m] - 1.0

            # Calculate delta
            # Avoid division by zero if any denom2 is 0 (aa[k] could be 0? No, init with >0)
            # But aa[m]-1 could be 0 if aa[m]=1. We checked aa[m]==1 above.
            delta = V21 / denom1 - V11 / denom2

            # [~, q] = max(delta)
            q = np.argmax(delta)

            if m != q:
                # Update
                aa[q] += 1
                aa[m] -= 1

                ybby[m] = V11[m]
                ybby[q] = V21[q]

                # Update Y
                Y[i, m] = 0
                Y[i, q] = 1
                label[i] = q

                # Update BBUU
                # BBUU(:,m) = BBUU(:,m) - BBB(:,i)
                # BBUU(:,q) = BBUU(:,q) + BBB(:,i)

                # BBB[:, i] access
                if issparse(BBB):
                    # Efficient column slicing for CSC
                    BBB_i_col = BBB[:, i].toarray().flatten()
                else:
                    BBB_i_col = BBB[:, i]

                BBUU[:, m] -= BBB_i_col
                BBUU[:, q] += BBB_i_col

        iter_num += 1

        # Compute objective function value
        current_obj = np.sum(ybby / aa)
        obj_max.append(current_obj)

    Y_label = label
    return Y_label, iter_num, np.array(obj_max)
