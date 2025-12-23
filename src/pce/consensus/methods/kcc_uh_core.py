from typing import List

import numpy as np

from .KCC_TMS_2023.RunKCC import RunKCC


def kcc_uh_core(
        BPi: np.ndarray,
        n_clusters: int,
        weights: np.ndarray,
        u_type: List[str] = ['u_h', 'std'],
        rep: int = 5,
        max_iter: int = 100,
        minThres: float = 1e-5,
        util_flag: int = 0,
) -> np.ndarray:
    """
    Core implementation of KCC_UH (K-means Consensus Clustering with Harmonic Utility).

    Parameters
    ----------
    BPi : np.ndarray
        Base Partitions matrix (n_samples, n_estimators).
    n_clusters : int
        Target number of clusters.
    weights : np.ndarray
        Weights for each base partition.
    u_type : list
        Type of utility function. Defaults to ['u_h', 'std'] for KCC_UH.
    rep : int
        Number of times the k-means algorithm will be run with different centroid seeds.
    max_iter : int
        Maximum number of iterations of the k-means algorithm.
    minThres : float
        Convergence threshold.
    util_flag : int
        Utility flag.

    Returns
    -------
    labels : np.ndarray
        Consensus cluster labels.
    """
    # Note: KCC_UH uses u_type=['u_h', 'std'] passed from the wrapper or default here.
    pi_sumbest, pi_index, pi_converge, pi_utility, t = RunKCC(BPi, n_clusters, u_type, weights, rep, max_iter, minThres, util_flag)
    return pi_index
