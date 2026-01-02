import time
from typing import Optional, List

import numpy as np

from .methods.mdechc_core import mdechc_core
from .utils.get_k_target import get_k_target


def mdechc(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    MDECHC (Multi-Diversity Ensemble Clustering via Hierarchical Clustering) Wrapper.
    Corresponds to the main logic of MATLAB script run_MDECHC_TCYB_2022.m.

    The algorithm logic:
    1. Slice base clusterers (BPs)
    2. Calculate all Segments (getAllSegs)
    3. Calculate ECI (getECI)
    4. Calculate consensus matrix S (getLWCA)
    5. Perform Hierarchical Clustering (performHC)

    Note on Consistency with MATLAB:
    MATLAB script generates a fixed list of seeds derived from the master seed.
    This implementation replicates that behavior to ensure reproducibility per repetition.

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
    # Handle MATLAB's 1-based indexing
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

    # Generate random seed pool
    # MATLAB: random_seeds = randi([0, 1000000], 1, nRepeat);
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
        # Step B: Run MDECHC Core
        # -------------------------------------------------
        t_start = time.time()

        try:
            # Set specific seed for current iteration
            current_seed = random_seeds[iRepeat]
            np.random.seed(current_seed)

            # Call encapsulated core logic
            # Note: MDECHC core logic needs nBase to calculate LWCA
            label_pred = mdechc_core(BPi, nCluster, nBase)

            # Ensure output is a flattened numpy array
            final_label = np.array(label_pred).flatten()

        except Exception as e:
            print(f"MDECHC failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
