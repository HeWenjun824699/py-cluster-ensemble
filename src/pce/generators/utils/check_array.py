import numpy as np
import scipy.sparse as sp


def check_array(X, dtype=np.float64, accept_sparse=False):
    """
    Unified processing of input data X:
    1. If the algorithm does not support sparse matrices (accept_sparse=False), automatically convert to dense numpy array.
    2. Ensure the data type is correct.
    """
    # If it is a sparse matrix and the algorithm does not accept sparse matrices, convert it
    if sp.issparse(X):
        if not accept_sparse:
            # This is a memory-sensitive operation; for very large datasets, it might need to be decided by the user,
            # but for a general-purpose utility library, for ease of use, it typically defaults to converting to dense
            X = X.toarray()

    # Ensure it is a numpy array (if originally list or tuple)
    if not isinstance(X, (np.ndarray, sp.spmatrix)):
        X = np.array(X)

    return X.astype(dtype, copy=False)
