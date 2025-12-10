import numpy as np


def balance_evl(k, N_cluster):
    """
    Evaluate the balance of the distribution of the clustering.
    
    Parameters:
    k : int
        Number of clusters
    N_cluster : array-like
        Vector containing number of samples in each cluster
    
    Returns:
    entropy, bal, stDev, RME
    """
    N_cluster = np.array(N_cluster)
    aa = np.zeros(k)
    bb = np.zeros(k)
    
    N = np.sum(N_cluster)
    
    # In MATLAB: for i=1:k. 
    # Warning: N_cluster might not have length k if some clusters are empty and not passed in?
    # The caller usually constructs N_cluster.
    # We assume N_cluster is indexable up to k-1 or we loop over what we have.
    # MATLAB code: for i=1:k, accesses N_cluster(i).
    # So N_cluster must have length >= k.
    
    for i in range(k):
        # Handle potential index error if N_cluster is shorter than k
        if i < len(N_cluster):
            Ni = N_cluster[i]
        else:
            Ni = 0
            
        Ni = Ni + np.finfo(float).eps # eps
        
        a = (Ni / N) * np.log(Ni / N)
        aa[i] = a
        
        b = (Ni - N / k)**2
        bb[i] = b
        
    entropy = -1 / np.log(k) * np.sum(aa)
    stDev = (1 / (k - 1) * np.sum(bb))**(1/2)
    
    # Min/Max logic
    # Filter out eps if it was added solely for log? 
    # MATLAB: bal = min(N_cluster)/max(N_cluster). 
    # It uses the input N_cluster directly.
    # If N_cluster has zeros, min is 0.
    
    if len(N_cluster) > 0:
        # We should use the original values for min/max logic, not the loop values with eps?
        # The MATLAB code uses N_cluster(i) inside loop for calc, 
        # but bal = min(N_cluster)/max(N_cluster) uses the vector.
        bal = np.min(N_cluster) / np.max(N_cluster)
        RME = np.min(N_cluster) / (N / k)
    else:
        bal = 0
        RME = 0

    return entropy, bal, stDev, RME
