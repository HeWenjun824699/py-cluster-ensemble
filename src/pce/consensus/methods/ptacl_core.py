import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from .PTAAL_PTACL_PTASL_PTGP_TKDE_2016.computeMicroclusters import compute_microclusters
from .PTAAL_PTACL_PTASL_PTGP_TKDE_2016.computeMCA import compute_mca
from .PTAAL_PTACL_PTASL_PTGP_TKDE_2016.computePTS_fast_v3 import compute_pts_fast_v3
from .PTAAL_PTACL_PTASL_PTGP_TKDE_2016.mapMicroclustersBackToObjects import map_microclusters_back_to_objects


def ptacl_core(base_cls, n_cluster):
    """
    Produce microclusters and run PTA-CL.
    Corrected logic: Directly clusters the PTS matrix, removing the redundant
    and buggy intermediate step found in the original MATLAB file.
    """
    # 1. Compute Microclusters
    mc_base_cls, mc_labels = compute_microclusters(base_cls)
    tilde_n = mc_base_cls.shape[0]

    # 2. Compute MCA
    mca = compute_mca(mc_base_cls)

    # 3. Set Parameters
    para = {'K': np.floor(np.sqrt(tilde_n) / 2), 'T': np.floor(np.sqrt(tilde_n) / 2)}

    # 4. Compute PTS (Probability Trajectory Similarity)
    pts = compute_pts_fast_v3(mca, mc_labels, para)

    # 5. Run Hierarchical Clustering (Complete Linkage) DIRECTLY on PTS
    # Correction: Directly call local CL clustering instead of calling the problematic run_pta_sl_local
    if np.isscalar(n_cluster):
        ks = [n_cluster]
    else:
        ks = n_cluster

    # Directly pass PTS for clustering
    mc_results_cl = run_pta_cl_local(pts, ks)

    # 6. Map back
    label_cl = map_microclusters_back_to_objects(mc_results_cl, mc_labels)

    return label_cl


def run_pta_cl_local(S, ks):
    """
    Corresponds to function [results_cl] = runPTA_CL(S, ks) in PTACL.m
    Performs Complete Linkage clustering on the Similarity Matrix S.
    """
    n = S.shape[0]

    # Convert Similarity to Distance
    d = stod2_local(S)

    # Hierarchical Clustering (Complete Linkage)
    zcl = linkage(d, method='complete')

    results_cl = np.zeros((n, len(ks)), dtype=int)
    for i, k in enumerate(ks):
        # fcluster usually returns 1..k
        results_cl[:, i] = fcluster(zcl, k, criterion='maxclust')

    return results_cl


def stod2_local(S):
    """
    Converts similarity matrix to condensed distance vector.
    """
    n = S.shape[0]
    # Use numpy efficient operations instead of loops to improve large matrix performance
    # Get the upper triangular matrix part (excluding diagonal)
    # scipy.cluster.hierarchy.linkage requires condensed distance matrix
    # Order is: (0,1), (0,2)...(0,n), (1,2)...

    # To maintain strict consistency with Matlab stod2 (taking upper triangle row by row)
    s_vec = []
    for i in range(n - 1):
        s_vec.append(S[i, i + 1:])

    s = np.concatenate(s_vec)
    d = 1.0 - s
    return d
