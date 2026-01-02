import time
from typing import Optional

import numpy as np

from .methods.mcla_core import mcla_core
from .utils.get_k_target import get_k_target


def mcla(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    MCLA (Meta-Clustering Algorithm) Wrapper.
    Corresponds to the main logic of MATLAB script: Batch read BPs, slice and run MCLA, evaluate and save results.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators)
    Y : np.ndarray, optional
        True labels, used to infer the number of clusters k
    nClusters : int, optional
        Target number of clusters k
    nBase : int, default=20
        Number of base clusterers used in each repeated experiment
    nRepeat : int, default=10
        Number of experiment repetitions
    seed : int, default=2026
        Random seed

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
        A tuple containing:
        - labels_list : A list of predicted labels (np.ndarray) for each repetition.
        - time_list   : A list of execution times (float) for each repetition.
    """

    # 1. Process data
    # [Critical] Handle MATLAB's 1-based indexing
    # MCLA core algorithm usually needs 0-based indexing to build hypergraph or matrix
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # --- [Modification] Call helper function to get unique K value ---
    # One line solution, reuse logic
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. Experiment loop
    # Prepare result container
    labels_list = []
    time_list = []

    # Initialize random number generator
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # Step A: Slice BPs
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
        # Step B: Run MCLA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        t_start = time.time()

        try:
            # Call core algorithm
            # Note: Python implementation usually directly receives (n_samples, n_estimators)
            # But mcla_core (ported from MATLAB) expects (n_clusterings, n_samples)
            label_pred = mcla_core(BPi.T, nCluster)
            label_pred = np.array(label_pred).flatten()
        except Exception as e:
            print(f"MCLA failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros_like(Y)

        labels_list.append(label_pred)
        t_cost = time.time() - t_start
        time_list.append(t_cost)

    return labels_list, time_list
