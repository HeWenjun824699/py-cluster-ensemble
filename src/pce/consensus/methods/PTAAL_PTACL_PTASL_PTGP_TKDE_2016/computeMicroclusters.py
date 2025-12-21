import numpy as np

def compute_microclusters(base_cls):
    """
    Obtain the set of microclusters w.r.t. the clustering ensemble.
    
    Args:
        base_cls: N x M matrix of cluster labels.
        
    Returns:
        new_base_cls: Unique rows of base_cls (microclusters).
        m_cls_labels: Array of shape (N, 2), where col 0 is original index, 
                      and col 1 is microcluster label (1-based index).
    """
    # Find unique rows and the inverse mapping
    # np.unique returns sorted unique elements of an array.
    # axis=0 ensures we look at unique rows.
    # return_inverse=True returns the indices of the unique array that reconstruct the original array.
    new_base_cls, inverse_indices = np.unique(base_cls, axis=0, return_inverse=True)
    
    # Construct mClsLabels to match Matlab output format: [original_index, microcluster_label]
    # Matlab uses 1-based indexing for the label.
    n_samples = base_cls.shape[0]
    m_cls_labels = np.zeros((n_samples, 2), dtype=int)
    m_cls_labels[:, 0] = np.arange(n_samples) # 0-based index for python, but logical equivalent
    m_cls_labels[:, 1] = inverse_indices + 1  # 1-based microcluster index
    
    return new_base_cls, m_cls_labels
