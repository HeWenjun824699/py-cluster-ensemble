import numpy as np


def cltoclb(cl):
    # cl is expected to be a 1D array of cluster labels.
    n_samples = len(cl)
    max_label = int(np.max(cl))
    
    # Create binary matrix (one-hot encoding style)
    # Rows correspond to labels 0..max_label
    clb = np.zeros((max_label + 1, n_samples))
    
    for i in range(max_label + 1):
        # Mark samples that belong to cluster i
        clb[i, :] = (cl == i).astype(float)
        
    return clb
