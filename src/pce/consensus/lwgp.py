import time
from typing import Optional, List

import numpy as np

from .methods.lwgp_core import lwgp_core
from .utils.get_k_target import get_k_target


def lwgp(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        theta: float = 10,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Locally weighted ensemble clustering (LWGP).

    LWGP is a graph-based ensemble method that incorporates local weighting
    to refine the similarity structure. It represents the ensemble as a
    bipartite graph and utilizes efficient graph partitioning techniques to
    resolve the final consensus partition.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels for target cluster count k inference.
    nClusters : int, optional
        Target number of clusters k.
    theta : float, default=10
        Weighting threshold parameter (t) regulating the local confidence
        of base clusters.
    nBase : int, default=20
        Number of base clusterers per repetition.
    nRepeat : int, default=10
        Number of experiment repetitions.
    seed : int, default=2026
        Seed for ensuring consistent results across multiple runs.

    Returns
    -------
    labels_list : list of np.ndarray
        Predicted consensus labels for each repetition.
    time_list : list of float
        Computation time cost for each run.


    .. note:: **Source**

        Huang et al., "Locally weighted ensemble clustering", *TCYB*, 2017.

        **BibTeX**

        .. code-block:: bibtex

            @article{huang2017locally,
                title={Locally weighted ensemble clustering},
                author={Huang, Dong and Wang, Chang-Dong and Lai, Jian-Huang},
                journal={IEEE transactions on cybernetics},
                volume={48},
                number={5},
                pages={1460--1473},
                year={2017},
                publisher={IEEE}
            }
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
        # Step B: Run LWGP
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # Explicitly set the global seed to match MATLAB's logic inside the loop
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # Call core algorithm
            # MATLAB: label = LWGP_v1(BPi, nCluster, t);
            label_pred = lwgp_core(BPi, nCluster, theta)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"LWGP failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
