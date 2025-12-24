import numpy as np

def l2_distance_1(a, b):
    """
    Compute squared Euclidean distance.
    ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
    
    Parameters:
    a, b : numpy.ndarray
        Matrices where each COLUMN is a data point (matching MATLAB logic).
        
    Returns:
    d : numpy.ndarray
        Distance matrix.
    """
    # MATLAB: if (size(a,1) == 1) a = [a; zeros(1,size(a,2))]; end
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
    if b.shape[0] == 1:
        b = np.vstack([b, np.zeros((1, b.shape[1]))])
        
    # MATLAB: aa=sum(a.*a); bb=sum(b.*b); ab=a'*b;
    # Sum over rows (dim 0), resulting in row vector
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = np.dot(a.T, b)
    
    # MATLAB: d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;
    # aa' is (n, 1), bb is (1, m)
    # Python broadcasting handles repmat automatically
    # aa[:, None] is (n, 1)
    # bb[None, :] is (1, m)
    d = aa[:, np.newaxis] + bb[np.newaxis, :] - 2 * ab
    
    # MATLAB: d = real(d); d = max(d,0);
    d = np.real(d)
    d = np.maximum(d, 0)
    
    return d
