import time
from typing import Optional, List

import numpy as np

from .methods.trce_core import trce_core
from .utils.get_k_target import get_k_target


def trce(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        gamma: float = 0.01,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Tri-level Robust Clustering Ensemble with Multiple Graph Learning (TRCE).

    TRCE formulates the consensus clustering problem as a multiple graph learning
    task. It constructs a unified framework that integrates structural information
    from three levels: instance-level constraints, cluster-level associations, and
    instance-cluster relationships. By learning a consensus graph from these
    multi-view structures, TRCE achieves high robustness against noise and
    inconsistent base partitions.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels for k inference.
    nClusters : int, optional
        Target number of clusters k.
    gamma : float, default=0.01
        Regularization parameter controlling the trade-off between the data
        fidelity and the smoothness of the consensus matrix.
    nBase : int, default=20
        Number of base clusterers processed per slice.
    nRepeat : int, default=10
        Number of independent repetitions.
    seed : int, default=2026
        Master seed ensuring identical experimental conditions across runs.

    Returns
    -------
    labels_list : list of np.ndarray
        Prediction results for `nRepeat` independent runs.
    time_list : list of float
        Time cost for each tensor-based optimization run.


    .. note:: **Source**

        Zhou et al., "Tri-level Robust Clustering Ensemble with Multiple Graph Learning", *AAAI*, 2021.

        **BibTeX**

        .. code-block:: bibtex

            @inproceedings{zhou2021tri,
                title={Tri-level robust clustering ensemble with multiple graph learning},
                author={Zhou, Peng and Du, Liang and Shen, Yi-Dong and Li, Xuejun},
                booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
                volume={35},
                number={12},
                pages={11125--11133},
                year={2021}
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
        # Step B: Run TRCE
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # Explicitly set the global seed to match MATLAB's logic inside the loop
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # Call core algorithm
            # MATLAB logic:
            # 1. Ai construction (Tensor)
            # 2. S = optimization(Ai, nCluster, gamma)
            # 3. label = conncomp(graph(S))

            # Assume trce_core encapsulates Ai construction, optimization, and conncomp
            label_pred = trce_core(BPi, nCluster, gamma)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"TRCE failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
