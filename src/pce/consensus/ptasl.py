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
    PTASL (Probability Trajectory based Association with Single Linkage) Wrapper.
    Corresponds to the main logic of MATLAB script run_PTASL_TKDE_2016.m.

    The algorithm flow is as follows (based on TKDE 2016 paper):
    1. Generate Microclusters
    2. Calculate Microcluster Co-Association Matrix (MCA)
    3. Calculate Probability Trajectory Similarity (PTS)
    4. Use Single Linkage for final clustering

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
