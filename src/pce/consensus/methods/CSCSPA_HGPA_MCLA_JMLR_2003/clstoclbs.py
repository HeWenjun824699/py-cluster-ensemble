import numpy as np
from cltoclb import cltoclb

def clstoclbs(cls):
    # cls: (n_clusterings, n_samples)
    clbs = None
    
    # Iterate over each clustering (row)
    for i in range(cls.shape[0]):
        # Get binary representation for this clustering
        lb = cltoclb(cls[i, :])
        if clbs is None:
            clbs = lb
        else:
            # Concatenate vertically
            clbs = np.vstack((clbs, lb))
            
    return clbs
