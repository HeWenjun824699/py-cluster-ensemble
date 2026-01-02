import time
from typing import Optional, List

import numpy as np

from .methods.ptgp_core import ptgp_core
from .utils.get_k_target import get_k_target


def ptgp(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Probability Trajectory based Graph Partitioning (PTGP).

    PTGP formulates the ensemble problem as a graph partitioning task. It
    extracts micro-clusters and builds a similarity graph based on probability
    trajectories (PTS). The final consensus is obtained by partitioning this
    graph using spectral clustering or similar graph-cut techniques.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels used for target cluster count inference.
    nClusters : int, optional
        Target number of clusters k.
    nBase : int, default=20
        Number of base partitions used per repetition slice.
    nRepeat : int, default=10
        Number of experimental repetitions.
    seed : int, default=2026
        Global seed for reproducible BPs slicing and internal graph partitioning.

    Returns
    -------
    labels_list : list of np.ndarray
        A list of prediction results for `nRepeat` repetition experiments.
    time_list : list of float
        Execution time (seconds) for each repetition.
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
    # Generate nRepeat random seeds (corresponds to MATLAB: randi([0, 1000000], 1, nRepeat))
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
        # Step B: Run PTGP
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # CRITICAL FIX: Explicitly set the global seed to match MATLAB's rng(random_seeds(iRepeat))
        # This ensures KMeans and other stochastic steps start from the exact same state as MATLAB's loop.
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # Call core algorithm
            label_pred = ptgp_core(BPi, nCluster)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"PTGP failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
