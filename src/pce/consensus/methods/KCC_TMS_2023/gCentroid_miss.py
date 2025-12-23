import numpy as np
from .sCentroid_miss import sCentroid_miss

def gCentroid_miss(IDX, index, K, n, r, sumKi, Ki):
    """
    Update centroid for missing data.
    """
    total_cols = int(sumKi[-1])
    C = np.zeros((K, total_cols))
    num = np.zeros(K)
    maxKi = int(np.max(Ki))
    
    for k in range(K):
        members = (index == k)
        
        if np.any(members):
            num[k] = np.sum(members)
            idx_members = IDX[members, :]
            
            # Count including missing (value -1)
            # We map -1 -> 0, 0 -> 1... for counting
            shifted_members = idx_members + 1
            
            # Check if all members in a partition are missing
            for i in range(r):
                # bincount on column i
                # bins: 0 (missing), 1 (cluster 0), 2 (cluster 1)...
                counts = np.bincount(shifted_members[:, i], minlength=int(Ki[i])+1)
                
                missing_count = counts[0]
                cluster_counts = counts[1:int(Ki[i])+1]
                
                if missing_count == num[k]:
                    # All missing
                    rand_c = np.random.randint(Ki[i])
                    C[k, int(rand_c + sumKi[i])] = 1
                else:
                    valid_num = num[k] - missing_count
                    if valid_num > 0:
                         C[k, int(sumKi[i]):int(sumKi[i+1])] = cluster_counts / valid_num
                    else:
                        # Should be covered by missing_count == num[k], but just in case
                        rand_c = np.random.randint(Ki[i])
                        C[k, int(rand_c + sumKi[i])] = 1
        else:
            rand_idx = np.random.randint(n)
            sample_row = IDX[rand_idx, :].reshape(1, -1)
            C[k, :] = sCentroid_miss(sample_row, 1, r, Ki, sumKi)
            
    return C
