import numpy as np

def map_microclusters_back_to_objects(results, mc_labels):
    """
    Map microcluster labels back to original objects.
    
    Args:
        results: Clustering results on microclusters (N_micro x N_results).
        mc_labels: Mapping (N_objects x 2), col 1 is microcluster index (1-based).
        
    Returns:
        full_results: Clustering results for original objects (N_objects x N_results).
    """
    # Sort mc_labels by original index (col 0)
    # Python sorts are stable, but let's be explicit
    # mc_labels col 0 is just 0..N-1 usually, but let's sort to be safe
    # sort_idx = np.argsort(mc_labels[:, 0])
    # mc_labels2 = mc_labels[sort_idx]
    # Actually, if mc_labels[:, 0] is just index, we can just use it directly.
    # But following Matlab logic:
    
    # In Python, if we assume mc_labels corresponds to row indices 0..N-1:
    micro_indices = mc_labels[:, 1] - 1 # Convert to 0-based
    
    # results is (N_micro, cntRes)
    # We want full_results (N_objects, cntRes)
    # full_results[i, :] = results[micro_indices[i], :]
    
    full_results = results[micro_indices, :]
    
    return full_results
