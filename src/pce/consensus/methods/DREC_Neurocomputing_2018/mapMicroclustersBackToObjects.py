import numpy as np

def map_microclusters_back_to_objects(results, mc_labels):
    """
    Map microcluster results back to original objects.
    
    Parameters:
    results (numpy.ndarray): Clustering results on microclusters (N_micro x cntRes) or vector
    mc_labels (numpy.ndarray): (N_objects x 2) array mapping original indices to microcluster IDs.
                               Col 1: Original Index (1-based from computeMicroclusters)
                               Col 2: Microcluster ID (1-based from computeMicroclusters)
    
    Returns:
    numpy.ndarray: Full results mapped back to objects (N_objects x cntRes)
    """
    
    # MATLAB: [~, I] = sort(mcLabels(:,1));
    # mcLabels2 = mcLabels(I,:);
    # In our Python computeMicroclusters, mc_labels is already sorted by original index (row 0 to N-1).
    # But for safety, we respect the input.
    
    # Sort by first column (original index)
    # argsort returns indices to sort the array.
    sort_indices = np.argsort(mc_labels[:, 0])
    mc_labels2 = mc_labels[sort_indices]
    
    N = mc_labels.shape[0]
    
    # Check if results is 1D or 2D
    if results.ndim == 1:
        results = results.reshape(-1, 1)
        
    cnt_res = results.shape[1]
    full_results = np.zeros((N, cnt_res))
    
    # for i = 1:cntRes
    #    fullResults(:,i) = results(mcLabels2(:,2),i);
    
    # mcLabels2[:, 1] contains 1-based microcluster IDs.
    # Python requires 0-based indices for 'results'.
    # So we subtract 1.
    
    microcluster_indices = mc_labels2[:, 1].astype(int) - 1
    
    for i in range(cnt_res):
        # results rows correspond to microclusters.
        # We pick the row corresponding to the microcluster ID of each object.
        full_results[:, i] = results[microcluster_indices, i]
        
    return full_results
