import numpy as np

def compute_MCA_jyh(BP):
    """
    Compute MCA matrix (Micro-Cluster Association?) 
    Checks if rows of BP are identical (samples belong to same clusters in ALL partitions).
    
    Args:
        BP: Base partitions matrix (n_samples x m_partitions)
        
    Returns:
        MCA_ML: Matrix where (i,j) is 1 if sample i and j have identical label vectors.
    """
    # n= size(BP,1); % number of samples
    # m=size(BP,2); % number of the BPs
    n, m = BP.shape
    
    # MCA_ML=zeros(n);
    # for i=1:n
    #     for j=1:n
    #         if BP(i,:)==BP(j,:)
    #             MCA_ML(i,j)=1;
    #         end
    #     end
    # end
    
    # Optimized Python implementation using broadcasting
    # We compare the entire rows.
    # (BP[:, None, :] == BP[None, :, :]) creates (n, n, m) boolean tensor
    # .all(axis=2) checks if all elements in the m-dimension are True
    
    MCA_ML = (BP[:, np.newaxis, :] == BP[np.newaxis, :, :]).all(axis=2).astype(float)
    
    return MCA_ML
