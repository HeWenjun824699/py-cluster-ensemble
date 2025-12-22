import numpy as np

def L2_distance_1(a, b):
    """
    Compute squared Euclidean distance
    ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
    
    Args:
        a: matrix (d x n1) - each column is a data point
        b: matrix (d x n2) - each column is a data point
        
    Returns:
        d: distance matrix (n1 x n2)
    """
    # Ensure inputs are 2D arrays
    if a.ndim == 1:
        a = a[:, np.newaxis]
    if b.ndim == 1:
        b = b[:, np.newaxis]

    # sum(a.*a) -> sum over columns (axis 0)
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = a.T @ b

    # d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab
    # In numpy broadcasting:
    # aa[:, None] is (n1, 1)
    # bb[None, :] is (1, n2)
    d = aa[:, np.newaxis] + bb[np.newaxis, :] - 2 * ab

    d = np.real(d)
    d = np.maximum(d, 0)

    return d
