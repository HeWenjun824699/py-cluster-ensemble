import time
from typing import Optional

import numpy as np

from .methods.hgpa_core import hgpa_core
from .utils.get_k_target import get_k_target


def hgpa(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026
) -> tuple[list[np.ndarray], list[float]]:
    """
    Cluster Ensembles --- A Knowledge Reuse Framework for Combining Multiple Partitions (HGPA).

    HGPA models the ensemble problem by representing each cluster in the base
    partitions as a hyperedge in a hypergraph. The algorithm then seeks an
    optimal partitioning of the hypergraph that minimizes the weight of cut
    hyperedges, effectively finding a consensus that preserves the cluster
    memberships across the ensemble.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators). Supports
        both 0-based and 1-based (MATLAB style) indexing.
    Y : np.ndarray, optional
        True labels of shape (n_samples,). Used to infer the target number
        of clusters if `nClusters` is not provided.
    nClusters : int, optional
        The target number of clusters for the final result. If provided,
        it overrides the inference from `Y`.
    nBase : int, default=20
        Number of base partitions used in a single ensemble experiment (slice size).
    nRepeat : int, default=10
        Number of independent repetitions. Total base partitions required
        is `nBase` * `nRepeat`.
    seed : int, default=2026
        Random seed for reproducibility of internal hypergraph partitioning
        initialization.

    Returns
    -------
    labels_list : list of np.ndarray
        A list containing `nRepeat` prediction arrays, each of shape (n_samples,).
    time_list : list of float
        A list containing the execution time (in seconds) for each repetition.


    .. note:: **Source**

        Strehl et al., "Cluster Ensembles --- A Knowledge Reuse Framework for Combining Multiple Partitions", *JMLR*, 2002.

        **BibTeX**

        .. code-block:: bibtex

            @article{strehl2002cluster,
                title={Cluster ensembles---a knowledge reuse framework for combining multiple partitions},
                author={Strehl, Alexander and Ghosh, Joydeep},
                journal={Journal of machine learning research},
                volume={3},
                number={Dec},
                pages={583--617},
                year={2002}
            }
    """

    # 1. Extract data (Load BPs and Y)
    # [Critical] Handle MATLAB's 1-based indexing
    # HGPA core algorithm usually based on hypergraph, needs 0-based indexing to build incidence matrix
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # --- [Modification] Call helper function to get unique K value ---
    # One line solution, reuse logic
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    # 2. Experiment loop
    # Prepare result container
    labels_list = []
    time_list = []

    # Initialize random number generator
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # Step A: Slice BPs
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
        # Step B: Run HGPA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        t_start = time.time()

        try:
            # Call core algorithm
            # Note: Kept .T style from mcla_old.py
            # If hgpa_core expects (n_samples, n_estimators), remove .T
            label_pred = hgpa_core(BPi.T, nCluster)
            label_pred = np.array(label_pred).flatten()
        except Exception as e:
            print(f"HGPA failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros_like(Y)

        labels_list.append(label_pred)
        t_cost = time.time() - t_start
        time_list.append(t_cost)

    return labels_list, time_list
