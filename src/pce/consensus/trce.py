import time
from typing import Optional, List

import numpy as np

from .methods.trce_core import trce_core
from .utils.get_k_target import get_k_target


def trce(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        gamma: float = 0.01,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    TRCE (Tensor-based Regularized Clustering Ensemble) Wrapper.
    Corresponds to the main logic of MATLAB script run_TRCE_AAAI_2021.m.

    The algorithm typically includes the following steps:
    1. Convert base clusterings to co-association matrix tensor (Ai)
    2. Solve tensor optimization problem to get consensus matrix S
    3. Build graph based on S and calculate connected components to get final clustering

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators)
    Y : np.ndarray, optional
        True labels, used to infer the number of clusters k
    nClusters : int, optional
        Target number of clusters k
    gamma : float, default=0.01
        Parameter gamma in TRCE algorithm.
        Corresponds to MATLAB code: gam = -2; gamma = 10.^gam (i.e., 0.01)
    nBase : int, default=20
        Number of base clusterers used in each repeated experiment
    nRepeat : int, default=10
        Number of experiment repetitions
    seed : int, default=2026
        Random seed (corresponds to seed = 2026 in MATLAB script)

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

    # Initialize random number generator (corresponds to MATLAB: rng(seed, 'twister'))
    rs = np.random.RandomState(seed)
    # Generate nRepeat random seeds (corresponds to MATLAB: random_seeds = randi([0, 1000000], 1, nRepeat))
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # Step A: Slice BPs (Get base clusterers for current round)
        # -------------------------------------------------
        # MATLAB logic: idx = (iRepeat - 1) * nBase + 1 : iRepeat * nBase;
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
        # Step B: Run TRCE
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # Explicitly set the global seed to match MATLAB's logic inside the loop
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # Call core algorithm
            # MATLAB logic:
            # 1. Ai construction (Tensor)
            # 2. S = optimization(Ai, nCluster, gamma)
            # 3. label = conncomp(graph(S))

            # Assume trce_core encapsulates Ai construction, optimization, and conncomp
            label_pred = trce_core(BPi, nCluster, gamma)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"TRCE failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
