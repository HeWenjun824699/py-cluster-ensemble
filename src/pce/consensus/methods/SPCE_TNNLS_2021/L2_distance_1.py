import numpy as np

def L2_distance_1(a, b):
    """
    L2_distance_1 computes squared Euclidean distance.
    ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
    
    a, b: matrices where each column is a data point.
    """
    
    # Ensure inputs are 2D arrays
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
        
    # Replicate MATLAB logic for 1-row inputs
    # if (size(a,1) == 1) ...
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
    if b.shape[0] == 1:
        b = np.vstack([b, np.zeros((1, b.shape[1]))])

    # aa = sum(a.*a) -> sum columns
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = np.dot(a.T, b)
    
    # d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;
    # Broadcasting: (n_a, 1) + (1, n_b) - (n_a, n_b)
    d = aa[:, np.newaxis] + bb[np.newaxis, :] - 2 * ab
    
    d = np.real(d)
    d = np.maximum(d, 0)
    
    return d
