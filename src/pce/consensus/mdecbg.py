import time
from typing import Optional, List, Tuple

import numpy as np

from .methods.mdecbg_core import mdecbg_core
from .utils.get_k_target import get_k_target


def mdecbg(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Toward Multi-Diversified Ensemble Clustering of High-Dimensional Data: From Subspaces to Metrics and Beyond (MDECBG).

    MDECBG is a representative algorithm of the Multi-Diversity Ensemble
    Clustering (MDEC) framework. It integrates diverse base partitions by
    constructing segments and employing Entropy-based Consensus Information
    (ECI) to weight a bipartite graph. The final results are produced by
    efficient bipartite graph partitioning.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels used to infer k if `nClusters` is None.
    nClusters : int, optional
        Target number of clusters k.
    nBase : int, default=20
        Number of base clusterers used in each experimental repetition.
    nRepeat : int, default=10
        Number of independent repetitions.
    seed : int, default=2026
        Random seed to maintain reproducibility of slicing and graph partitioning.

    Returns
    -------
    labels_list : list of np.ndarray
        List of ensemble prediction results for each run.
    time_list : list of float
        List of execution times for each run in seconds.


    .. note:: **Source**

        Huang et al., "Toward Multi-Diversified Ensemble Clustering of High-Dimensional Data: From Subspaces to Metrics and Beyond", *TCYB*, 2021.

        **BibTeX**

        .. code-block:: bibtex

            @article{huang2021toward,
                title={Toward multidiversified ensemble clustering of high-dimensional data: From subspaces to metrics and beyond},
                author={Huang, Dong and Wang, Chang-Dong and Lai, Jian-Huang and Kwoh, Chee-Keong},
                journal={IEEE Transactions on Cybernetics},
                volume={52},
                number={11},
                pages={12231--12244},
                year={2021},
                publisher={IEEE}
            }
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
    # Generate nRepeat seeds for each iteration here
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
        # Step B: Run MDECBG
        # -------------------------------------------------
        t_start = time.time()

        try:
            # Set specific seed for current iteration
            # MATLAB: rng(random_seeds(iRepeat));
            current_seed = random_seeds[iRepeat]
            np.random.seed(current_seed)

            # Call encapsulated core logic
            label_pred = mdecbg_core(BPi, nCluster)

            # Ensure output is a flattened numpy array
            final_label = np.array(label_pred).flatten()

        except Exception as e:
            print(f"MDECBG failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            final_label = np.zeros(nSmp, dtype=int)

        labels_list.append(final_label)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
