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
) -> tuple[list[np.ndarray], list[float]]:
    """
    Ensemble Clustering via Co-association Matrix Self-enhancement (ECCMS).

    ECCMS implements a self-enhancement mechanism for the co-association matrix.
    It utilizes Entropy-based Consensus Information (ECI) and selects high-confidence
    sample pairs to refine the similarity structure before performing spectral
    clustering for the final consensus.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels used to determine k if nClusters is not provided.
    nClusters : int, optional
        Target number of clusters k.
    alpha : float, default=0.8
        Threshold for High Confidence (HC) matrix generation, controlling the
        selection of reliable sample pairs.
    lamb : float, default=0.4
        Regularization parameter for Entropy-based Consensus Information (ECI)
        calculation and spectral clustering weights.
    nBase : int, default=20
        Number of base clusterings processed per repetition.
    nRepeat : int, default=10
        Number of independent repetitions.
    seed : int, default=2026
        Global seed for reproducible BPs slicing and internal spectral clustering.

    Returns
    -------
    labels_list : list of np.ndarray
        Prediction results for `nRepeat` experimental runs.
    time_list : list of float
        Execution time (seconds) for each run.
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
    time_list = []

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
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
