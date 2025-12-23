import numpy as np
from .sCentroid import sCentroid

def gCentroid(IDX, index, K, n, r, sumKi, Ki):
    """
    Update centroid for each cluster.
    
    Args:
        IDX: (n, r) matrix of labels (0-based).
        index: (n,) vector of assignments (0-based).
        K: number of clusters.
        n: number of points.
        r: number of partitions.
        sumKi: offset vector.
        Ki: cluster counts vector.
    """
    total_cols = int(sumKi[-1])
    C = np.zeros((K, total_cols))
    num = np.zeros(K)
    maxKi = int(np.max(Ki))
    
    for k in range(K):
        members = (index == k)
        
        if np.any(members):
            num[k] = np.sum(members)
            idx_members = IDX[members, :] # (num_k, r)
            
            # MATLAB: counts = hist(idx, 1:maxKi)
            # This counts occurrences of 1..maxKi.
            # Python: idx_members has 0..maxKi-1.
            # We want counts for each partition i.
            
            # Optimization: Process per partition to match MATLAB logic:
            # if size(counts, 1) == 1: C(...) = 1 
            # (This logic in MATLAB handles single member case where hist returns row vector?)
            # Actually if num[k] == 1, idx_members is (1, r).
            
            if num[k] == 1:
                # idx_members is (1, r)
                # indices = idx_members + sumKi[:r]
                # C[k, indices] = 1
                indices = idx_members.flatten() + sumKi[:r]
                C[k, indices.astype(int)] = 1
            else:
                for i in range(r):
                    # count occurrences of 0..Ki[i]-1
                    c = np.bincount(idx_members[:, i], minlength=int(Ki[i]))
                    # c might be longer than Ki[i] if maxKi > Ki[i], but labels shouldn't exceed Ki[i]-1
                    
                    # MATLAB: C(k, sumKi(i)+1:sumKi(i+1)) = counts(1:Ki(i), i)' / num(k)
                    # Python indices: sumKi[i] : sumKi[i+1]
                    
                    # Ensure c is truncated or padded to length Ki[i]
                    count_vec = c[:int(Ki[i])]
                    
                    start_idx = int(sumKi[i])
                    end_idx = int(sumKi[i+1])
                    
                    C[k, start_idx:end_idx] = count_vec / num[k]
                    
        else:
            # Empty cluster, re-initialize randomly
            # randsample(n, 1) -> random index
            rand_idx = np.random.randint(n)
            # Make sure to pass a 2D array to sCentroid: (1, r)
            sample_row = IDX[rand_idx, :].reshape(1, -1)
            C[k, :] = sCentroid(sample_row, 1, r, sumKi)
            
    return C
