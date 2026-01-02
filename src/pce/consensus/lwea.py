import time
from typing import Optional, List

import numpy as np

from .methods.lwea_core import lwea_core
from .utils.get_k_target import get_k_target


def lwea(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        theta: float = 10,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Locally Weighted Ensemble Clustering (LWEA).

    LWEA introduces a local weighting strategy to evaluate the reliability of
    clusters in base partitions. It constructs a weighted bipartite graph
    representing the relationships between samples and clusters, and solves the
    consensus partition via graph partitioning or spectral methods.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators). Supports
        automatic conversion from 1-based indexing.
    Y : np.ndarray, optional
        True labels used to infer the number of clusters k if `nClusters`
        is not provided.
    nClusters : int, optional
        Target number of clusters k.
    theta : float, default=10
        Threshold parameter (t in the paper) for calculating local weights,
        controlling the influence of cluster-wise reliability.
    nBase : int, default=20
        Number of base clusterers used in each experimental repetition slice.
    nRepeat : int, default=10
        Number of independent repetitions for statistical evaluation.
    seed : int, default=2026
        Random seed for reproducibility of slicing and internal solvers.

    Returns
    -------
    labels_list : list of np.ndarray
        A list of predicted label arrays for `nRepeat` repetitions.
    time_list : list of float
        A list of execution times for each repetition in seconds.
    """

    # 1. Data preprocessing
    # Handle MATLAB's 1-based indexing (if min is 1, subtract 1)
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # Get target number of clusters
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. Experiment loop configuration
    labels_list = []
    time_list = []

    # Initialize random number generator (corresponds to MATLAB: rng(seed, 'twister'))
    rs = np.random.RandomState(seed)
    # Generate nRepeat random seeds (corresponds to MATLAB: random_seeds = randi([0, 1000000], 1, nRepeat))
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # Step A: Slice BPs (Get base clusterers for current round)
        # -------------------------------------------------
        start_idx = iRepeat * nBase
        end_idx = (iRepeat + 1) * nBase

        # Boundary check
        if start_idx >= nTotalBase:
            print(f"Warning: Not enough Base Partitions for repeat {iRepeat + 1}")
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        BPi = BPs[:, start_idx:end_idx]

        # -------------------------------------------------
        # Step B: Run LWEA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # Explicitly set the global seed to match MATLAB's logic inside the loop
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # Call core algorithm
            # MATLAB: label = LWEA_v1(BPi, nCluster, t);
            label_pred = lwea_core(BPi, nCluster, theta)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"LWEA failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
