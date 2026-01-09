import time
import numpy as np
from typing import Optional, List, Tuple, Union
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def sc3(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2026,
        return_matrix: bool = False
) -> Union[Tuple[List[np.ndarray], List[float]],
           Tuple[List[np.ndarray], List[float], np.ndarray]]:
    """
    SC3: consensus clustering of single-cell RNA-seq data (SC3).

    SC3 integrates a subset of base partitions into a co-association matrix
    and derives the final partition via Complete Linkage Hierarchical Clustering.
    This approach ensures a stable and robust consensus result by capturing
    high-confidence structural similarities across the ensemble.

    Parameters
    ----------
    BPs : numpy.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators).
    Y : numpy.ndarray, optional
        True labels used to determine the target cluster count if ``nClusters`` is None.
        Default is None.
    nClusters : int, optional
        The target number of clusters (k). If None, it is inferred from ``Y``.
        Default is None.
    nBase : int, default=20
        The number of base partitions used to construct the consensus matrix per repeat.
    nRepeat : int, default=10
        The number of repetition experiments.
    seed : int, default=2026
        Random seed for selecting base partitions.
    return_matrix : bool, default=False
        Whether to return the last computed consensus matrix.

    Returns
    -------
    labels_list : list of numpy.ndarray
        A list of predicted labels for each repetition (0-based indexing).
    time_list : list of float
        A list of time costs (in seconds) for each ensemble run.
    cons_mat : np.ndarray, optional
        The consensus matrix from the last repetition, returned only if return_matrix is True.


    .. note:: **Source**

        Kiselev et al., "SC3: consensus clustering of single-cell RNA-seq data", *Nature Methods*, 2017.

        **BibTeX**

        .. code-block:: bibtex

            @article{kiselev2017sc3,
                title={SC3: consensus clustering of single-cell RNA-seq data},
                author={Kiselev, Vladimir Yu and Kirschner, Kristina and Schaub, Michael T and Andrews, Tallulah and Yiu, Andrew and Chandra, Tamir and Natarajan, Kedar N and Reik, Wolf and Barahona, Mauricio and Green, Anthony R and others},
                journal={Nature methods},
                volume={14},
                number={5},
                pages={483--486},
                year={2017},
                publisher={Nature Publishing Group US New York}
            }
    """

    # 1. Preprocessing
    # Adjust 1-based indexing to 0-based for internal consistency check (optional)
    # Note: consensus_matrix logic works regardless of 0 or 1 based as long as equality holds
    n_samples = BPs.shape[0]
    n_total_base = BPs.shape[1]

    # Determine target cluster count
    # Assuming get_k_target logic is: if nClusters, use it; else use len(unique(Y))
    if nClusters is None:
        if Y is not None:
            nClusters = len(np.unique(Y))
        else:
            raise ValueError("nClusters must be specified if Y is not provided.")

    # 2. Experiment Loop Setup
    labels_list = []
    time_list = []
    cons_mat = None

    rs = np.random.RandomState(seed)
    # Seeds for selection
    repeat_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        t_start = time.time()

        # -------------------------------------------------
        # Step A: Select Base Partitions
        # -------------------------------------------------
        # Option 1: Sequential slicing (like CDEC example)
        # start_idx = iRepeat * nBase
        # end_idx = (iRepeat + 1) * nBase
        # if end_idx > n_total_base: break
        # subset_bps = BPs[:, start_idx:end_idx]

        # Option 2: Random sampling (Standard for Ensemble stability check)
        # matches the 'seed' parameter intent better
        current_rng = np.random.RandomState(repeat_seeds[iRepeat])
        indices = current_rng.choice(n_total_base, nBase, replace=False)
        subset_bps = BPs[:, indices]

        try:
            # -------------------------------------------------
            # Step B: Compute Consensus Matrix
            # -------------------------------------------------
            cons_mat = _consensus_matrix(subset_bps)

            # -------------------------------------------------
            # Step C: Hierarchical Clustering (SC3 Logic)
            # -------------------------------------------------
            # SC3 uses Euclidean distance on the consensus matrix rows
            # pdist calculates pairwise distances between observations
            cons_dists = pdist(cons_mat, metric='euclidean')

            # Complete Linkage
            Z = linkage(cons_dists, method='complete')

            # Cut Tree
            # criterion='maxclust' ensures we get exactly nClusters
            labels = fcluster(Z, t=nClusters, criterion='maxclust')

            # Convert to 0-based indexing for Python consistency
            final_label = labels - 1

        except Exception as e:
            print(f"SC3 Consensus failed on repeat {iRepeat}: {e}")
            final_label = np.zeros(n_samples, dtype=int)

        labels_list.append(final_label)
        time_list.append(time.time() - t_start)

    if return_matrix:
        return labels_list, time_list, cons_mat
    else:
        return labels_list, time_list


def _consensus_matrix(partitions: np.ndarray) -> np.ndarray:
    """
    Calculate the consensus matrix from base partitions.

    Parameters
    ----------
    partitions : numpy.ndarray
        The cluster labels matrix of shape (n_samples, n_partitions).

    Returns
    -------
    consensus : numpy.ndarray
        The consensus matrix of shape (n_samples, n_samples).
    """
    n_samples = partitions.shape[0]
    n_partitions = partitions.shape[1]

    consensus = np.zeros((n_samples, n_samples))

    for i in range(n_partitions):
        labels = partitions[:, i]
        # Outer equality check
        # (N, 1) == (1, N) -> (N, N) bool matrix
        mat = (labels[:, None] == labels[None, :])
        consensus += mat.astype(float)

    consensus /= n_partitions
    return consensus
