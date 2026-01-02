import time
from typing import Optional, List

import numpy as np

from .methods.ptaal_core import ptaal_core
from .utils.get_k_target import get_k_target


def ptaal(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Probability Trajectory based Association for Active Learning (PTAAL).

    PTAAL utilizes probability trajectories to model the relationships between
    samples and clusters. It is designed to work within an active learning
    scenario or as a standalone consensus function that leverages the
    association-based probability trajectories of samples across base
    partitions.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators). Supports
        both 0-based and 1-based (MATLAB style) indexing.
    Y : np.ndarray, optional
        True labels of shape (n_samples,). Used to infer the target number
        of clusters k if `nClusters` is not provided.
    nClusters : int, optional
        The target number of clusters for the final result.
    nBase : int, default=20
        Number of base partitions used in each experimental repetition slice.
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
    # Generate nRepeat random seeds (corresponds to MATLAB: randi([0, 1000000], 1, nRepeat))
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # Step A: Slice BPs
        # -------------------------------------------------
        # MATLAB: idx = (iRepeat - 1) * nBase + 1 : iRepeat * nBase;
        start_idx = iRepeat * nBase
        end_idx = (iRepeat + 1) * nBase

        # Boundary check
        if start_idx >= nTotalBase:
            print(f"Warning: Not enough Base Partitions for repeat {iRepeat + 1}")
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        # Get current subset of base clusterings
        BPi = BPs[:, start_idx:end_idx]

        # -------------------------------------------------
        # Step B: Run PTAAL
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        t_start = time.time()

        try:
            # Call core algorithm
            label_pred = ptaal_core(BPi, nCluster)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"PTAAL failed on repeat {iRepeat}: {e}")
            # Return all-zero labels as placeholder on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
