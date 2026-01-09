import time
from typing import Optional, List, Tuple

import numpy as np

from .methods.kcc_uh_core import kcc_uh_core
from .utils.get_k_target import get_k_target


def kcc_uh(
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
) -> tuple[list[np.ndarray], list[float]]:
    """
    Algorithm 1038: KCC: A MATLAB Package for k-Means-based Consensus Clustering (KCC_UH).

    KCC_UH maximizes the Harmonic Utility function by transforming the ensemble
    problem into a weighted K-means task. It is designed to be robust and
    computationally efficient by leveraging the relationship between consensus
    clustering and the K-means objective.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators). Supports
        automatic conversion from 1-based indexing.
    Y : np.ndarray, optional
        True labels for cluster count inference.
    nClusters : int, optional
        Target number of clusters k.
    rep : int, default=5
        Number of internal random restarts (n_init) per ensemble run.
    max_iter : int, default=100
        Maximum iterations for the KCC optimization core.
    min_thres : float, default=1e-5
        Convergence threshold for stopping the iteration.
    util_flag : int, default=0
        Algorithm-specific flag for utility feature calculations.
    nBase : int, default=20
        Number of base partitions used per repetition slice.
    nRepeat : int, default=10
        Number of independent repetitions for statistical evaluation.
    seed : int, default=2026
        Global seed ensuring reproducible slicing and optimization starts.

    Returns
    -------
    labels_list : list of np.ndarray
        List of ensemble result arrays for `nRepeat` repetitions.
    time_list : list of float
        Computation time cost for each repetition.


    .. note:: **Source**

        Lin et al., "Algorithm 1038: KCC: A MATLAB Package for k-Means-based Consensus Clustering", *TMS*, 2023.

        **BibTeX**

        .. code-block:: bibtex

            @article{lin2023algorithm,
                title={Algorithm 1038: KCC: A MATLAB Package for k-Means-based Consensus Clustering},
                author={Lin, Hao and Liu, Hongfu and Wu, Junjie and Li, Hong and G{\"u}nnemann, Stephan},
                journal={ACM transactions on mathematical software},
                volume={49},
                number={4},
                pages={1--27},
                year={2023},
                publisher={ACM New York, NY}
            }
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
            print(f"KCC_UH Warning: Not enough Base Partitions for repeat {iRepeat + 1}")
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

            # KCC_UH Specific: Use 'u_h' (Harmonic Utility)
            u_type = ['u_h', 'std']

            # Call the core logic
            # Matches MATLAB: [~, label] = RunKCC(BPi, nCluster, U, w, rep, maxIter, minThres, utilFlag);
            label_pred = kcc_uh_core(
                BPi=BPi,
                n_clusters=nCluster,
                weights=w,
                u_type=u_type,
                rep=rep,
                max_iter=max_iter,
                minThres=min_thres,
                util_flag=util_flag
            )

            # Flatten to 1D array
            final_label = np.array(label_pred).flatten().astype(int)

        except Exception as e:
            print(f"KCC_UH failed on repeat {iRepeat}: {e}")
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"KCC_UH Repeat {iRepeat + 1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
