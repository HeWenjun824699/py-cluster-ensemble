import time
from typing import Optional, List

import numpy as np

from .methods.gtlec_core import gtlec_core
from .utils.get_k_target import get_k_target


def gtlec(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        alpha: float = 0.05,
        beta: float = 7.0,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    GTLEC (Graph-Based Tensor Learning for Ensemble Clustering) Wrapper.
    Replicates the logic of run_GTLEC_MM_2023.m.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels, used to determine k if nClusters is not provided.
    nClusters : int, optional
        Target number of clusters.
    alpha : float, default=0.05
        Parameter for TensorEC (as seen in MATLAB script).
    beta : float, default=7.0
        Parameter for TensorEC (as seen in MATLAB script).
    nBase : int, default=20
        Number of base partitions to use per repetition.
    nRepeat : int, default=10
        Number of experimental repetitions.
    seed : int, default=2026
        Master random seed.

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
        A tuple containing:
        - labels_list : A list of predicted labels (np.ndarray) for each repetition.
        - time_list   : A list of execution times (float) for each repetition.
    """

    # 1. Preprocessing
    # Adjust 1-based indexing (MATLAB) to 0-based (Python) if necessary
    if np.min(BPs) >= 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # Determine target cluster count
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. Experiment Loop Setup
    labels_list = []
    time_list = []

    # Initialize Random State (matches MATLAB's rng(seed, 'twister'))
    rs = np.random.RandomState(seed)

    # Generate seeds for each repetition (matches MATLAB: randi([0, 1000000], 1, nRepeat))
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # Step A: Slice BPs (Select Base Partitions)
        # -------------------------------------------------
        start_idx = iRepeat * nBase
        end_idx = (iRepeat + 1) * nBase

        # Boundary checks
        if start_idx >= nTotalBase:
            print(f"GTLEC Warning: Not enough Base Partitions for repeat {iRepeat + 1}")
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        # Extract current subset of base partitions
        BPi = BPs[:, start_idx:end_idx]

        # -------------------------------------------------
        # Step B: Run GTLEC Core
        # -------------------------------------------------
        t_start = time.time()

        try:
            # Set specific seed for this iteration
            current_seed = random_seeds[iRepeat]
            np.random.seed(current_seed)

            # Call the core logic
            # Corresponds to: [S,Z,obj]=TensorEC(...); label=SpectralClustering(...);
            label_pred = gtlec_core(BPi, nCluster, alpha, beta)

            # Flatten to 1D array
            final_label = np.array(label_pred).flatten()

        except Exception as e:
            print(f"GTLEC failed on repeat {iRepeat}: {e}")
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"GTLEC Repeat {iRepeat + 1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
