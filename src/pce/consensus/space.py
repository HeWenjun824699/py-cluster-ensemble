import time
from typing import Optional, List

import numpy as np

from .methods.space_core import space_core
from .utils.get_k_target import get_k_target


def space(
        BPs: np.ndarray,
        Y: np.ndarray,
        nClusters: Optional[int] = None,
        gamma: float = 4.0,
        batch_size: int = 50,
        delta: float = 0.1,
        n_active_rounds: int = 10,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Sequential Probing and Active Clustering Ensemble (SPACE).

    SPACE is an advanced ensemble framework that incorporates active learning
    and sequential probing. It adaptively selects and queries informative sample
    relationships from base partitions to refine the consensus, significantly
    reducing the labeling effort required for high-quality clustering.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray
        True labels, required for the active probing and selection process.
    nClusters : int, optional
        Target number of clusters k.
    gamma : float, default=4.0
        Hyperparameter regulating the influence of the sequential probing logic.
    batch_size : int, default=50
        The number of samples or relationships queried in each active round.
    delta : float, default=0.1
        Threshold parameter for the internal optimization solver.
    n_active_rounds : int, default=10
        Number of active learning iterations for sequential probing.
    nBase : int, default=20
        Number of base clusterers used per ensemble repetition.
    nRepeat : int, default=10
        Number of experiment repetitions.
    seed : int, default=2026
        Global seed ensuring reproducible slicing and stochastic active rounds.

    Returns
    -------
    labels_list : list of np.ndarray
        Consensus partitions for each experimental repetition.
    time_list : list of float
        Computation time cost for each run in seconds.
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
        # MATLAB: idx = (iRepeat - 1) * nBase + 1 : iRepeat * nBase;
        start_idx = iRepeat * nBase
        end_idx = (iRepeat + 1) * nBase

        # Boundary checks
        if start_idx >= nTotalBase:
            print(f"SPACE Warning: Not enough Base Partitions for repeat {iRepeat + 1}")
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        # Extract current subset of base partitions
        BPi = BPs[:, start_idx:end_idx]

        # -------------------------------------------------
        # Step B: Run SPACE Core
        # -------------------------------------------------
        t_start = time.time()

        try:
            # Set specific seed for this iteration
            # MATLAB: rng(random_seeds(iRepeat));
            current_seed = random_seeds[iRepeat]
            np.random.seed(current_seed)

            # Call the core logic
            label_pred = space_core(
                BPi=BPi, 
                nCluster=nCluster, 
                Y=Y, 
                gamma=gamma, 
                batch_size=batch_size, 
                delta=delta,
                n_active_rounds=n_active_rounds
            )

            # Flatten to 1D array
            final_label = np.array(label_pred).flatten()

        except Exception as e:
            print(f"SPACE failed on repeat {iRepeat}: {e}")
            # Fallback to zeros
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"SPACE Repeat {iRepeat + 1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
