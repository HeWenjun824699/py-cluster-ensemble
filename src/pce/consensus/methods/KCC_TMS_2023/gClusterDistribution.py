import numpy as np

def gClusterDistribution(IDX, Ki, n):
    """
    Calculates cluster distribution for basic partitions.
    
    Args:
        IDX: (n, r) matrix of cluster labels (0-based).
        Ki: (r,) vector, number of clusters in each partition.
        n: number of data points.
        
    Returns:
        P: (max(Ki), r) matrix, cluster distribution.
    """
    r = IDX.shape[1]
    maxKi = int(np.max(Ki))
    P = np.zeros((maxKi, r))
    
    for i in range(r):
        # bincount counts occurrences of non-negative integers. 
        # minlength ensures we get counts up to maxKi-1 (if we used maxKi as size)
        # But here we want up to maxKi items.
        # Assuming IDX contains labels 0 to Ki[i]-1.
        counts = np.bincount(IDX[:, i], minlength=maxKi)
        # P = counts / (n - counts) according to matlab code: 
        # P = counts(2:maxKi+1,:)./repmat(n-counts(1,:),maxKi,1); 
        # Wait, the MATLAB code does `hist(IDX, 0:maxKi)`. 
        # If IDX has 1..K, `hist` with 0..K bins will count 0s in bin 1, 1s in bin 2.
        # The matlab code: `counts(2:maxKi+1,:)`. This implies it discards the count of 0s.
        # And divides by `n - counts(1,:)`. counts(1,:) is the count of 0s.
        # So it calculates probability of cluster assignment GIVEN it's not missing (0).
        # If IDX is 0-based (0..K-1), and -1 indicates missing?
        # The MATLAB code supports missing values (0).
        # In Python, let's assume -1 or NaN for missing, but the input IDX here 
        # passed from Preprocess is likely handled. 
        
        # However, gClusterDistribution in MATLAB:
        # maxKi = max(Ki);
        # counts = hist(IDX,0:maxKi); 
        # P = counts(2:maxKi+1,:)./repmat(n-counts(1,:),maxKi,1);
        
        # If input IDX (Python) has 0..K-1 for clusters and -1 for missing.
        # We need to replicate this logic.
        
        c = np.zeros(maxKi + 1)
        # Shift IDX by +1 so -1 becomes 0, 0 becomes 1...
        # If IDX is just 0-based valid labels:
        # We treat them as 1-based to match the logic or just handle logic directly.
        
        # Let's check where it's called. Preprocess.m calls it.
        # If `missFlag==1`, IDX has 0s.
        # If `missFlag==0`, IDX has no 0s (labels 1..K).
        
        # Python port: we will expect IDX to be 0-based for valid clusters.
        # Missing values should be handled. If the caller passes -1 for missing.
        
        # Let's count -1 as index 0, 0 as index 1, etc.
        current_col = IDX[:, i]
        # Treat -1 as 0 (missing), 0 as 1 (cluster 1), etc.
        shifted_col = current_col + 1 
        c = np.bincount(shifted_col, minlength=maxKi + 1)
        
        # c[0] is count of missing. c[1:] are counts of clusters.
        # P column: c[1:] / (n - c[0])
        
        if n - c[0] > 0:
            P[:, i] = c[1:maxKi+1] / (n - c[0])
        else:
            P[:, i] = 0 # Should not happen if n > missing
            
    return P
