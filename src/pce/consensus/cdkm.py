import time
from typing import Optional, List, Tuple

import numpy as np

from .methods.cdkm_core import compute_Hc, Y_Initialize
from .methods.cdkm_core import cdkm_core
from .utils.get_k_target import get_k_target


def cdkm(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        nInnerRepeat: int = 5,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    CDKM (Consensus Clustering via Discrete Kernel K-Means) Wrapper.
    Corresponds to the main logic of MATLAB script run_CDKM_TPAMI_2022.m.

    The algorithm contains a double-layer loop structure:
    1. Outer loop: Slice base clusterers (BPs)
    2. Inner loop: Run CDKM_fast with multiple initializations, take the result corresponding to the maximum objective function

    Note on Consistency with MATLAB:
    The provided MATLAB implementation of 'Y_Initialize' contains a hardcoded random seed (rng(2024)), 
    which causes the inner loop initialization to be identical across iterations. 
    This Python implementation intentionally diverges from that specific behavior by respecting 
    the varying seeds generated in the wrapper, allowing for proper exploration of initializations 
    as likely intended by the algorithm design.

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
        Number of experiment repetitions (outer loop)
    nInnerRepeat : int, default=5
        Number of inner loop repetitions, used for selection (corresponds to MATLAB logic: n_inner_repeat = 5)
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
    # MATLAB: random_seeds = randi([0, 1000000], 1, nRepeat * nRepeat);
    # Note: MATLAB code generates nRepeat*nRepeat seeds, but the logic when indexing is:
    # (iRepeat-1) * nRepeat + inner_repeat
    # To maintain logical consistency, generate the same number of seeds here
    total_seeds_needed = nRepeat * nRepeat
    random_seeds = rs.randint(0, 1000001, size=total_seeds_needed)

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
        # Step B: Run CDKM
        # -------------------------------------------------
        t_start = time.time()

        try:
            # 1. Construct hypergraph incidence matrix Hc
            # MATLAB: Hc = compute_Hc(BPi);
            Hc, _ = compute_Hc(BPi)

            best_label = np.zeros(nSmp, dtype=int)
            obj_all = -float('inf')

            t_start = time.time()

            # 2. Inner loop selection
            # MATLAB: for inner_repeat = 1:n_inner_repeat
            for inner_repeat in range(nInnerRepeat):
                # Determine seed for current inner loop
                # MATLAB logic: rng(random_seeds( (iRepeat-1) * nRepeat + inner_repeat ));
                # Note: Python indexing starts from 0, MATLAB starts from 1, need to align carefully
                seed_idx = iRepeat * nRepeat + inner_repeat
                if seed_idx >= len(random_seeds):
                    # fallback if inner logic exceeds pre-gen seeds
                    current_seed = rs.randint(0, 1000001)
                else:
                    current_seed = random_seeds[seed_idx]

                # Set global seed for current iteration
                np.random.seed(current_seed)

                # Initialize
                # MATLAB: [~, label_0] = Y_Initialize(nSmp, nCluster);
                _, label_0 = Y_Initialize(nSmp, nCluster)

                # Core optimization
                # MATLAB: [label, iter_num, obj_max] = CDKM_fast(Hc', label_0, nCluster);
                # Note: MATLAB passes Hc' (transpose).
                # Assume Python's cdkm_fast handles dimensions internally, or we transpose here.
                # Usually sklearn style data is (n_samples, n_features).
                # If Hc is (n_samples, n_hyperedges), no transpose needed.
                # If Hc is (n_hyperedges, n_samples), Hc.T is needed.
                # Here pass Hc, cdkm_core decides how to handle.
                label_pred, _, obj_history = cdkm_core(Hc.T, label_0, nCluster)

                # Get final objective function value
                # MATLAB: if obj_max(end) > obj_all
                current_obj = obj_history[-1] if isinstance(obj_history, (list, np.ndarray)) else obj_history

                if current_obj > obj_all:
                    obj_all = current_obj
                    best_label = label_pred.copy()

            # Ensure output is a flattened numpy array
            final_label = np.array(best_label).flatten()

        except Exception as e:
            print(f"CDKM failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
