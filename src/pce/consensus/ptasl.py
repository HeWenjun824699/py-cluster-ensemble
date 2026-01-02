import time
from typing import Optional, List

import numpy as np

from .methods.ptasl_core import ptasl_core
from .utils.get_k_target import get_k_target


def ptasl(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Probability Trajectory based Association with Single Linkage (PTASL).

    PTASL is a hierarchical consensus method that uses probability trajectory
    similarity (PTS) in conjunction with single linkage. Similar to its variants,
    it maps samples to micro-clusters and analyzes their movement across
    partitions to calculate a refined similarity matrix for clustering.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels used for target cluster count inference.
    nClusters : int, optional
        Target number of clusters k.
    nBase : int, default=20
        Number of base clusterers used per slice.
    nRepeat : int, default=10
        Number of independent experimental repetitions.
    seed : int, default=2026
        Master seed ensuring identical experimental conditions.

    Returns
    -------
    labels_list : list of np.ndarray
        Predicted consensus labels for `nRepeat` independent runs.
    time_list : list of float
        Execution time cost for each single linkage run.
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

    # Initialize random number generator
    rs = np.random.RandomState(seed)
    # Generate nRepeat random seeds
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
        # Step B: Run PTASL
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        t_start = time.time()

        try:
            # Call core algorithm
            label_pred = ptasl_core(BPi, nCluster)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"PTASL failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
