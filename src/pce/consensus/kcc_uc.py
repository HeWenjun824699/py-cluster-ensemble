import time
from typing import Optional, List, Tuple

import numpy as np

from .methods.kcc_uc_core import kcc_uc_core
from .utils.get_k_target import get_k_target


def kcc_uc(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        rep: int = 5,
        max_iter: int = 100,
        min_thres: float = 1e-5,
        util_flag: int = 0,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> List[np.ndarray]:
    """
    KCC_Uc (K-means Consensus Clustering with Utility) Wrapper.
    Replicates the logic of run_KCC_Uc_TMS_2023.m.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels, used to determine k if nClusters is not provided.
    nClusters : int, optional
        Target number of clusters.
    rep : int, default=5
        Number of internal repetitions (n_init) for the KCC core.
    max_iter : int, default=100
        Maximum iterations for the KCC optimization.
    min_thres : float, default=1e-5
        Convergence threshold.
    util_flag : int, default=0
        Flag for utility calculation.
    nBase : int, default=20
        Number of base partitions to use per repetition.
    nRepeat : int, default=10
        Number of experimental repetitions.
    seed : int, default=2026
        Master random seed.

    Returns
    -------
    labels_list : List[np.ndarray]
        List containing the consensus labels for each repetition.
    """

    # 1. Preprocessing
    # Ensure BPs is integer (crucial for np.bincount in core logic)
    BPs = BPs.astype(int)

    # Adjust 1-based indexing (MATLAB) to 0-based (Python) if necessary
    # Heuristic: If min is >= 1, it's definitely 1-based.
    # Note: If input has 0 and 0 represents 'missing', this heuristic won't catch it
    # because min will be 0. Users must convert 0-missing to -1 manually or ensures min>=1.
    if np.min(BPs) >= 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # Determine target cluster count
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. Experiment Loop Setup
    labels_list = []

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
            print(f"KCC_Uc Warning: Not enough Base Partitions for repeat {iRepeat + 1}")
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        # Extract current subset of base partitions
        BPi = BPs[:, start_idx:end_idx]

        # -------------------------------------------------
        # Step B: Run KCC Core
        # -------------------------------------------------
        t_start = time.time()

        try:
            # Set specific seed for this iteration
            current_seed = random_seeds[iRepeat]
            np.random.seed(current_seed)

            # Define Weights (MATLAB: w = ones(r,1))
            r = BPi.shape[1]
            w = np.ones(r)
            u_type = ['u_c', 'std']

            # Call the core logic
            # Matches MATLAB: [~, label] = RunKCC(BPi, nCluster, U, w, rep, maxIter, minThres, utilFlag);
            label_pred = kcc_uc_core(
                BPi=BPi,
                n_clusters=nCluster,
                u_type=u_type,
                weights=w,
                rep=rep,
                max_iter=max_iter,
                minThres=min_thres,
                util_flag=util_flag
            )

            # Flatten to 1D array
            final_label = np.array(label_pred).flatten().astype(int)

        except Exception as e:
            print(f"KCC_Uc failed on repeat {iRepeat}: {e}")
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        # print(f"KCC_Uc Repeat {iRepeat + 1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list
