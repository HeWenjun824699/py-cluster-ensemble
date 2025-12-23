from typing import List

import numpy as np

from .KCC_TMS_2023.RunKCC import RunKCC


def kcc_uc_core(
        BPi: np.ndarray,
        n_clusters: int,
        weights: np.ndarray,
        u_type: List[str] = ['u_c', 'std'],
        rep: int = 5,
        max_iter: int = 100,
        minThres: float = 1e-5,
        util_flag: int = 0,
) -> np.ndarray:
    """
    Core implementation of KCC (K-means Consensus Clustering).

    In the standard KCC framework, the consensus clustering problem is transformed
    into a K-means clustering problem over the base partitions.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix (n_samples, n_estimators).
    n_clusters : int
        Target number of clusters.
    weights : np.ndarray
        Weights for each base partition.
    u_type : tuple
        Type of utility function (reserved for future logic extension).
    rep : int
        Number of times the k-means algorithm will be run with different centroid seeds.
    max_iter : int
        Maximum number of iterations of the k-means algorithm.
    minThres : float
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers to declare convergence.
    util_flag : int
        Utility flag (reserved).
    random_state : int, optional
        Random seed.

    Returns
    -------
    labels : np.ndarray
        Consensus cluster labels.
    """
    pi_sumbest, pi_index, pi_converge, pi_utility, t = RunKCC(BPi, n_clusters, u_type, weights, rep, max_iter, minThres, util_flag)
    return pi_index