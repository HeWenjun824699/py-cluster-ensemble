import numpy as np
from scipy.sparse import csc_matrix

def Y_Initialize(n, c):
    """
    Initialize cluster labels ensuring each cluster has at least one sample.
    
    Parameters:
    n (int): Number of samples.
    c (int): Number of clusters.
    
    Returns:
    tuple: (Y, labels)
        Y: Indicator matrix (n x c), dense numpy array (as per full(Y) in MATLAB).
        labels: Label vector (n, 1) or (n,), 0-based.
    """
    # MATLAB: labels = 1:c;
    # Python 0-based: 0 to c-1
    labels_head = np.arange(c)
    
    # MATLAB: labels = [labels, randi(c, 1, n - c)];
    # randi(c) is 1..c. In Python randint(0, c) is 0..c-1
    if n > c:
        labels_tail = np.random.randint(0, c, size=n - c)
        labels = np.concatenate((labels_head, labels_tail))
    else:
        # Fallback if n <= c, though strictly the code assumes n > c usually
        labels = labels_head[:n]
        
    # MATLAB: labels = labels(randperm(n));
    # Shuffle the labels
    np.random.shuffle(labels)
    
    # MATLAB: Y = ind2vec(labels)'; Y = full(Y);
    # Create one-hot encoding
    # rows = 0..n-1
    # cols = labels
    rows = np.arange(n)
    cols = labels
    data = np.ones(n)
    
    # Create sparse then full
    Y_sparse = csc_matrix((data, (rows, cols)), shape=(n, c))
    Y = Y_sparse.toarray()
    
    # Return 0-based labels to be consistent with Python ecosystem
    return Y, labels
