import time
from typing import Optional, List

import numpy as np

from .methods.ecpcs_hc_core import ecpcs_hc_core
from .utils.get_k_target import get_k_target


def ecpcs_hc(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        theta: float = 20,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2024
) -> tuple[list[np.ndarray], list[float]]:
    """
    Ensemble Clustering by Propagation of Cluster-wise Similarities via Hierarchical Clustering (ECPCS-HC).

    This algorithm enhances consensus performance by propagating similarities
    between clusters across different base partitions. After constructing an
    enhanced similarity matrix, it applies Hierarchical Clustering to produce
    the final ensemble result.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels, used to infer the number of clusters k.
    nClusters : int, optional
        Target number of clusters k.
    theta : float, default=20
        Similarity propagation threshold (parameter t in the paper), controlling
        the intensity of similarity filtering between clusters.
    nBase : int, default=20
        Number of base clusterers used in each repeated experiment.
    nRepeat : int, default=10
        Number of experiment repetitions.
    seed : int, default=2024
        Random seed for reproducibility (default 2024 to match MATLAB reference).

    Returns
    -------
    labels_list : list of np.ndarray
        List of predicted labels for each experimental repetition.
    time_list : list of float
        List of execution times for each run in seconds.
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
        # Step B: Run ECPCS-HC
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # Explicitly set the global seed to match MATLAB's logic inside the loop
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # Call core algorithm
            # MATLAB: label = ECPCS_HC(BPi, t, nCluster);
            # Assume Python version core function directly returns label array
            # Note: Python parameter name uses theta to correspond to MATLAB's t
            label_pred = ecpcs_hc_core(BPi, theta, nCluster)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"ECPCS-HC failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
