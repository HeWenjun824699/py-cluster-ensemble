import numpy as np
from scipy.linalg import svd, norm
from sklearn.preprocessing import normalize
from computeMicroclusters import compute_microclusters
from Gbe import gbe
from Robust_LearnZ import robust_learn_z
from spectral_clustering import spectral_clustering
from mapMicroclustersBackToObjects import map_microclusters_back_to_objects

def DREC(E, K, lam):
    """
    Obtain Z by RLRR and perform clustering.
    
    Parameters:
    E (numpy.ndarray): base clustering (N x M)
    K (int): the number of clusters
    lam (float): regularization term (lambda)
    
    Returns:
    dict: Output containing 'Blable' (final labels)
    """
    
    # Input: E, base clustering
    #        K, the number of clusters
    #        lambda, regularization term

    # find Microclusters
    # [newBaseCls, mClsLabels] = computeMicroclusters(E);
    new_base_cls, m_cls_labels = compute_microclusters(E)

    # Binary data matrix
    # BE = Gbe(newBaseCls);
    BE = gbe(new_base_cls)
    
    # Z = Robust_LearnZ(BE',lambda);
    # BE is (N_micro x KM). BE' is (KM x N_micro).
    # Robust_LearnZ expects X.
    Z = robust_learn_z(BE.T, lam)

    # post processing
    # [U,S,V] = svd(Z,'econ');
    # MATLAB svd(Z, 'econ') for square matrix Z returns Z = U*S*V'.
    # Python svd(Z) returns u, s, vh.
    # If Z is N x N, 'econ' doesn't change much for full rank, but let's stick to standard svd.
    
    U, s, vh = np.linalg.svd(Z, full_matrices=False)
    # s is 1D array of singular values
    
    # S = diag(S); -> In MATLAB S is diagonal matrix. Python s is vector.
    # r = sum(S>1e-4*S(1));
    # Check max singular value s[0]
    r = np.sum(s > (1e-4 * s[0]))
    
    # U = U(:,1:r); S = S(1:r);
    U_r = U[:, :r]
    s_r = s[:r]
    
    # U = U*diag(sqrt(S));
    # In python, broadcasting U * sqrt(s) works if shapes align.
    # U is (N, r), sqrt(s) is (r,).
    # We want to multiply each column j of U by sqrt(s[j]).
    U_processed = U_r * np.sqrt(s_r)
    
    # U = normr(U);
    U_normalized = normalize(U_processed, axis=1, norm='l2')
    
    # L = (U*U').^4;
    # U*U' is (N x N).
    L = np.power(U_normalized @ U_normalized.T, 4)
    
    # Blable = spectral_clustering(L, K);
    # L is affinity matrix.
    b_label = spectral_clustering(L, K)
    
    # Blable = mapMicroclustersBackToObjects(Blable,mClsLabels);
    # b_label is 1D array. map_microclusters... expects (N, 1) or 1D.
    final_labels = map_microclusters_back_to_objects(b_label, m_cls_labels)
    
    return {'Blable': final_labels}
