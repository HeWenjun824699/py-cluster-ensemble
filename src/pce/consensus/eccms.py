import time
from typing import Optional, List

import numpy as np

from .methods.eccms_core import eccms_core
from .utils.get_k_target import get_k_target


def eccms(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        alpha: float = 0.8,
        lamb: float = 0.4,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> List[np.ndarray]:
    """
    ECCMS (Ensemble Clustering via Co-association Matrix and Spectral Clustering) Wrapper.
    Corresponding to the main logic of MATLAB script run_ECCMS_TNNLS_2023.m.

    Algorithm Logic:
    1. Slice Base Partitions (BPs).
    2. Compute Consensus via ECCMS Core (getAllSegs -> ECI -> LWCA -> HC -> Spectral).

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels, used to infer k if nClusters is not provided.
    nClusters : int, optional
        Target number of clusters k.
    alpha : float, default=0.8
        Hyperparameter for High Confidence (HC) matrix generation.
    lamb : float, default=0.4
        Hyperparameter for Entropy Calculation and Spectral Clustering.
    nBase : int, default=20
        Number of base clusterings to use per repetition.
    nRepeat : int, default=10
        Number of experiment repetitions.
    seed : int, default=2026
        Random seed for reproducibility.

    Returns
    -------
    labels_list : List[np.ndarray]
        List containing results for each repetition.
    """

    # 1. Data Preprocessing
    # Ensure 1-based indexing for internal Matlab-ported logic (getAllSegs expects 1-based)
    if np.min(BPs) == 0:
        BPs = BPs + 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # Get target cluster number
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. Experiment Loop Configuration
    labels_list = []

    # Initialize Random State (matches MATLAB: rng(seed, 'twister'))
    rs = np.random.RandomState(seed)

    # Generate seed pool
    # MATLAB: random_seeds = randi([0, 1000000], 1, nRepeat);
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # Step A: Slice BPs (Get current batch of base clusterings)
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
        # Step B: Run ECCMS Core
        # -------------------------------------------------
        t_start = time.time()

        try:
            # Set specific seed for this iteration
            current_seed = random_seeds[iRepeat]
            np.random.seed(current_seed)

            # Call core logic
            label_pred = eccms_core(
                BPi, 
                nCluster, 
                nBase, 
                alpha=alpha, 
                lamb=lamb
            )

            # Ensure output is flattened numpy array
            final_label = np.array(label_pred).flatten()

            # Convert back to 0-based if necessary (fcluster returns 1-based)
            if np.min(final_label) == 1:
                final_label = final_label - 1

        except Exception as e:
            print(f"ECCMS failed on repeat {iRepeat}: {e}")
            # Return zero labels on failure
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list
