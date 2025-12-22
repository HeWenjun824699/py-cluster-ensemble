import numpy as np

def compute_microclusters(base_cls):
    """
    Obtain the set of microclusters w.r.t. the clustering ensemble.
    
    Parameters:
    base_cls (numpy.ndarray): base clustering matrix
    
    Returns:
    tuple: (new_base_cls, m_cls_labels)
    """
    # Equivalent to [uniqueBaseCls, uI] = unique(baseCls,'rows') in MATLAB
    # MATLAB's unique(..., 'rows') sorts the output rows.
    # Numpy's unique(..., axis=0) sorts the output rows.
    # return_inverse=True gives the indices to reconstruct original array from unique rows.
    # unique_rows[inverse_indices] == base_cls
    
    unique_base_cls, inverse_indices = np.unique(base_cls, axis=0, return_inverse=True)
    
    # mClsLabels in MATLAB: col 1 is original index (I), col 2 is microcluster ID.
    # The MATLAB code logic seems to try to map original indices to microcluster IDs.
    # In Python, 'inverse_indices' IS the mapping from original row i to unique row index (microcluster ID).
    # inverse_indices has shape (N,).
    
    N = base_cls.shape[0]
    
    # Construct mClsLabels to match MATLAB output structure strictly: [original_index, microcluster_id]
    # MATLAB indices are 1-based.
    m_cls_labels = np.zeros((N, 2), dtype=int)
    m_cls_labels[:, 0] = np.arange(1, N + 1) # Original indices (1-based)
    m_cls_labels[:, 1] = inverse_indices + 1 # Microcluster IDs (1-based)
    
    # The MATLAB code has a complex block checking for 'flag_r2014a' and fixing labels.
    # This block essentially checks if the sorting/unique operation was stable or consistent.
    # With numpy's unique, we get a deterministic mapping.
    # We will assume numpy's result is correct and sufficient, skipping the workaround for MATLAB 2014a.
    
    new_base_cls = unique_base_cls
    
    return new_base_cls, m_cls_labels
