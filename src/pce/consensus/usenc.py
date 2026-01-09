import time
from typing import Optional, List

import numpy as np

from .methods.usenc_core import usenc_core
from .utils.get_k_target import get_k_target


def usenc(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Ultra-scalable spectral clustering and ensemble clustering (USENC).

    USENC addresses the scalability bottleneck in ensemble clustering by
    reformulating the consensus problem as a bipartite graph partitioning task.
    It constructs a bipartite graph between samples and base clusters to
    efficiently approximate spectral clustering, achieving near-linear time
    complexity and making it highly effective for large-scale datasets.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators). Supports
        automatic conversion from 1-based indexing.
    Y : np.ndarray, optional
        True labels used to infer the number of clusters k if `nClusters`
        is not provided.
    nClusters : int, optional
        Target number of clusters k for the final consensus result.
    nBase : int, default=20
        Number of base clusterers used in each experimental repetition slice.
    nRepeat : int, default=10
        Number of independent repetitions for statistical evaluation.
    seed : int, default=2026
        Random seed to ensure reproducibility of slicing and internal solvers.

    Returns
    -------
    labels_list : list of np.ndarray
        A list of predicted label arrays for `nRepeat` repetitions.
    time_list : list of float
        A list of execution times (in seconds) for each repetition.


    .. note:: **Source**

        Huang et al., "Ultra-scalable spectral clustering and ensemble clustering", *TKDE*, 2019.

        **BibTeX**

        .. code-block:: bibtex

            @article{huang2019ultra,
                title={Ultra-scalable spectral clustering and ensemble clustering},
                author={Huang, Dong and Wang, Chang-Dong and Wu, Jian-Sheng and Lai, Jian-Huang and Kwoh, Chee-Keong},
                journal={IEEE Transactions on Knowledge and Data Engineering},
                volume={32},
                number={6},
                pages={1212--1226},
                year={2019},
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
        # Step B: Run USENC
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # Explicitly set the global seed to match MATLAB's logic inside the loop
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # Call core algorithm
            # MATLAB: label = USENC_ConsensusFunction(BPi, nCluster);
            # Assume Python version core function directly returns label array
            label_pred = usenc_core(BPi, nCluster)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"USENC failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
