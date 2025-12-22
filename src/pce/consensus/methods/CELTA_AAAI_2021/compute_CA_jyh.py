import numpy as np

def compute_CA_jyh(BP):
    """
    Compute Co-Association matrix from Base Partitions.
    
    Args:
        BP: Base partitions matrix (n_samples x m_partitions)
            Each column represents a partition, values are cluster labels.
            
    Returns:
        CA: Co-Association matrix (n_samples x n_samples)
    """
    # n= size(BP,1); % number of samples
    # m=size(BP,2); % number of the BPs
    n, m = BP.shape
    
    # CA=zeros(n);
    CA = np.zeros((n, n))
    
    # for i=1:m
    for i in range(m):
        # v=BP(:,i);
        v = BP[:, i]
        
        # Optimized implementation of the nested loops:
        # s=zeros(n,max(v)); ... CA=CA+s*s';
        # s*s' is effectively checking if v(j) == v(k).
        # We can do this efficiently with broadcasting.
        
        # Create an adjacency matrix for the current partition
        # Element (j, k) is 1 if v[j] == v[k], else 0.
        # Note: Matlab code iterates up to max(v) to build 's' (indicator matrix), 
        # then computes s*s'. This results in a block diagonal matrix where 
        # blocks correspond to clusters.
        
        # Python broadcasting:
        # Compare column vector v with row vector v
        adjacency = (v[:, np.newaxis] == v[np.newaxis, :]).astype(float)
        
        CA += adjacency
        
    # CA=CA/m;
    CA = CA / m
    
    return CA
