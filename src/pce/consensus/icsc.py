import time
from typing import Optional, List, Tuple

import numpy as np
from sklearn import cluster

from .utils.get_k_target import get_k_target


def icsc(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026,
        n_init: int = 100,
        affinity: str = 'precomputed',
        assign_labels: str = 'discretize'
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Iterative consensus spectral clustering improves detection of subject and group level brain functional modules (ICSC).

    This function performs consensus clustering by iteratively selecting a subset of
    base partitions, computing a co-association matrix, and applying spectral clustering.

    Parameters
    ----------
    BPs : np.ndarray
        The base partitions matrix of shape (n_samples, n_base_partitions).
        Each column represents a base partition.
    Y : np.ndarray, optional
        The true labels of the samples, used to determine the target number of clusters
        if ``nClusters`` is not provided. Default is None.
    nClusters : int, optional
        The target number of clusters. If None, it is inferred from ``Y``.
        Default is None.
    nBase : int, optional
        The number of base partitions to use in each iteration. Default is 20.
    nRepeat : int, optional
        The number of iterations to repeat the process. Default is 10.
    seed : int, optional
        The random seed for reproducibility. Default is 2026.
    n_init : int, optional
        Number of time the k-means algorithm will be run with different centroid seeds
        in Spectral Clustering. Default is 100.
    affinity : str, optional
        How to construct the affinity matrix. Default is 'precomputed'.
    assign_labels : str, optional
        The strategy to use to assign labels in the embedding space.
        Options are 'kmeans', 'discretize', 'cluster_qr'. Default is 'discretize'.

    Returns
    -------
    Tuple[List[np.ndarray], List[float]]
        A tuple containing:
        - labels_list: A list of predicted labels for each iteration.
        - time_list: A list of execution times for each iteration.


    .. note:: **Source**

        Gupta et al., "Iterative consensus spectral clustering improves detection of subject and group level brain functional modules", *Scientific Reports*, 2020.

        **BibTeX**

        .. code-block:: bibtex

            @article{gupta2020iterative,
                title={Iterative consensus spectral clustering improves detection of subject and group level brain functional modules},
                author={Gupta, Sukrit and Rajapakse, Jagath C},
                journal={Scientific reports},
                volume={10},
                number={1},
                pages={7590},
                year={2020},
                publisher={Nature Publishing Group UK London}
            }
    """

    if np.min(BPs) == 1:
        BPs = BPs - 1
        
    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]
    n_target = get_k_target(n_clusters=nClusters, y=Y)
    
    labels_list = []
    time_list = []

    # Initialize random number generator
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)
    
    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # Step A: Slice BPs
        # -------------------------------------------------
        # Python: [start, end)
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
        # Step B: Compute Co-association Matrix (Key to fixing warnings)
        # -------------------------------------------------
        t_start = time.time()

        # 1. Compute co-association matrix (similarity matrix)
        similarity_matrix = _compute_coassociation_matrix(BPi)

        # 2. Add micro-noise (Smoothing)
        similarity_matrix += 1e-6

        # -------------------------------------------------
        # Step C: Run SpectralClustering
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]
        
        try:
            cl = cluster.SpectralClustering(
                n_clusters=n_target,
                random_state=current_seed,
                n_init=n_init,
                affinity=affinity,
                assign_labels=assign_labels
            )
            labels = cl.fit_predict(similarity_matrix)
        except Exception:
             labels = np.zeros(nSmp)
             
        t_end = time.time()
        
        labels_list.append(labels)
        time_list.append(t_end - t_start)
        
    return labels_list, time_list


def _compute_coassociation_matrix(BPi: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the Co-association Matrix.

    Parameters
    ----------
    BPi : np.ndarray
        A subset of base partitions of shape (n_samples, n_base).

    Returns
    -------
    np.ndarray
        The computed co-association matrix of shape (n_samples, n_samples),
        normalized to the range [0, 1].
    """
    n_samples, n_base = BPi.shape

    # Initialize matrix
    sim_mat = np.zeros((n_samples, n_samples))

    # Process column by column (each base clustering)
    for i in range(n_base):
        labels = BPi[:, i]
        # Broadcasting: if labels[a] == labels[b], then match_mat[a,b] = 1
        # shape: (N, 1) vs (1, N) -> (N, N)
        match_mat = (labels[:, None] == labels[None, :]).astype(float)
        sim_mat += match_mat

    # Normalize to [0, 1] interval
    sim_mat /= n_base

    return sim_mat
