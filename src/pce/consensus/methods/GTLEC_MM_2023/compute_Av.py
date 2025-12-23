import numpy as np

def compute_Av(BP):
    """
    input the base partition, where each column BP is a base partitions
    BP: (n_samples, n_base_partitions)
    """
    n, m = BP.shape
    CA = []
    
    for i in range(m):
        v = BP[:, i]
        # MATLAB: s=zeros(n,max(v));
        # Since v contains labels, and we assume they are 1-based (from MATLAB context),
        # max(v) is the number of clusters if they are sequential.
        # If they are 0-based in Python input, we need to adjust.
        # However, preserving strict logic:
        # If input BP is directly from MATLAB data, it likely has 1-based labels.
        # If so, max(v) is correct size.
        # If 0-based, max(v) is (num_clusters - 1).
        
        # But wait, s is (n, max(v)).
        # If v has label 'k', s(j, k) = 1.
        # If v is 0-based, k=0, s(j, 0)=1.
        # If v is 1-based, k=1, s(j, 1)=1.
        
        # We will assume the input BP might be 0-based or 1-based.
        # We need a matrix S such that S[j, label] = 1.
        # Size should be max(v) + 1 if 0-based? Or just max(v) if 1-based?
        # MATLAB code `s=zeros(n,max(v))` implies 1-based indexing for labels up to max(v).
        # We will dynamically size it to `int(np.max(v)) + 1` to be safe for 0-based or 1-based, 
        # but technically we only need `max(v)` columns if 1-based and we adjust index.
        
        # Let's check logic: s * s' is the co-association matrix.
        # The column dimension of s disappears in the product.
        # So we just need one-hot encoding.
        
        max_val = int(np.max(v))
        # If 1-based, max_val is e.g. 3. We need indices 0,1,2,3? Or just map 1->0?
        # To keep strictly consistent with "s(j,k)=1" where k comes from v:
        # If v has 1, then s needs index 1 valid.
        
        s = np.zeros((n, max_val + 1)) 
        
        for j in range(n):
            k = int(v[j])
            if k > 0 or (k == 0 and s.shape[1] > 0): # Handle 0 or 1 based
                s[j, k] = 1
                
        # If the original code assumed 1-based and tight packing (no 0), column 0 of s might be empty.
        # That's fine, it doesn't affect s @ s.T.
        
        CA.append(s @ s.T)
        
    return CA
