import numpy as np
from typing import Optional, Tuple


from .ECCMS_TNNLS_2023.getAllSegs import getAllSegs
from .ECCMS_TNNLS_2023.computeECI import computeECI
from .ECCMS_TNNLS_2023.computeLWCA import computeLWCA
from .ECCMS_TNNLS_2023.getCA import getCA
from .ECCMS_TNNLS_2023.getHC import getHC
from .ECCMS_TNNLS_2023.run_EC_CMS import run_EC_CMS


def eccms_core(
        BPi: np.ndarray,
        n_cluster: int,
        n_base: int,
        alpha: float = 0.8,
        lamb: float = 0.4
) -> np.ndarray:
    """
    Core implementation of the ECCMS algorithm iteration.
    Corresponds to the core loop logic in the MATLAB script run_ECCMS_TNNLS_2023.m.

    Translates the following MATLAB logic:
        [bcs, baseClsSegs] = getAllSegs(BPi);
        ECI = computeECI(bcs, baseClsSegs, lambda);
        LWCA = computeLWCA(baseClsSegs, ECI, nBase);
        CA = getCA(baseClsSegs, nBase);
        A = getHC(CA, alpha);
        label = run_EC_CMS(A, LWCA, nCluster, lambda);

    Parameters
    ----------
    BPi : np.ndarray
        Base partition results for the current round (Subset of Base Partitions), shape (n_samples, n_base).
    n_cluster : int
        Target number of clusters.
    n_base : int
        Number of base estimators.
    alpha : float, default=0.8
        Threshold parameter for the High Confidence (HC) matrix.
    lamb : float, default=0.4
        Parameter for ECI computation and spectral clustering.
        Note: Corresponds to lambda in MATLAB.

    Returns
    -------
    label : np.ndarray
        Predicted cluster labels, shape (n_samples,).
    """

    # 1. Get Segments and Basic Cluster Segments
    # MATLAB: [bcs, baseClsSegs] = getAllSegs(BPi);
    bcs, base_cls_segs = getAllSegs(BPi)

    # 2. Compute ECI (Entropy-based Consensus Information)
    # MATLAB: ECI = computeECI(bcs, baseClsSegs, lambda);
    eci = computeECI(bcs, base_cls_segs, lamb)

    # 3. Compute LWCA (Locally Weighted Co-Association)
    # MATLAB: LWCA = computeLWCA(baseClsSegs, ECI, nBase);
    lwca = computeLWCA(base_cls_segs, eci, n_base)

    # 4. Compute CA (Standard Co-Association)
    # MATLAB: CA = getCA(baseClsSegs, nBase);
    ca = getCA(base_cls_segs, n_base)

    # 5. Compute HC (High Confidence Matrix)
    # MATLAB: A = getHC(CA, alpha);
    A = getHC(ca, alpha)

    # 6. Run Final Spectral Clustering / NMF Solver
    # MATLAB: label = run_EC_CMS(A, LWCA, nCluster, lambda);
    label = run_EC_CMS(A, lwca, n_cluster, lamb)

    # 7. Post-processing
    # Ensure returning a 1D integer array
    label = np.array(label, dtype=int).flatten()

    # MATLAB code has logic: if min(label) == 0, label = label + 1
    # Python usually uses 0-based indexing, so if you want to maintain Python style (0 to k-1), you can return directly.
    # If run_EC_CMS returns 0-based labels internally and you need 1-based output perfectly consistent with MATLAB, adjust here.
    # Here, the original output of run_EC_CMS is maintained by default (0-based is generally recommended for Python).

    return label
