import time
from typing import Optional, Tuple, List, Union

import numpy as np
from sklearn.cluster import KMeans

from .utils.get_k_target import get_k_target


def dcc(
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
    Fine-grained subphenotypes in acute kidney injury populations based on deep clustering Derivation and interpretation (DCC).

    Parameters
    ----------
    BPs : np.ndarray
        Base partitions matrix of shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels, used only to infer the number of clusters if nClusters is not provided.
    nClusters : int, optional
        The target number of clusters. If None, it will be inferred from Y or default to valid range.
    nBase : int, default=20
        Number of base partitions to use in each repeat.
    nRepeat : int, default=10
        Number of repetitions (or chunks) to process.
    seed : int, default=2026
        Random seed for reproducibility.
    return_matrix : bool, default=False
        Whether to return the last computed consensus matrix.

    Returns
    -------
    labels_list : List[np.ndarray]
        List of predicted labels for each repetition.
    time_list : List[float]
        List of execution times for each repetition.
    M : np.ndarray, optional
        The consensus matrix from the last repetition, returned only if return_matrix is True.


    .. note:: **Source**

        Tan et al., "Fine-grained subphenotypes in acute kidney injury populations based on deep clustering Derivation and interpretation", *Scientific Reports*, 2024.

        **BibTeX**

        .. code-block:: bibtex

            @article{tan2024fine,
                title={Fine-grained subphenotypes in acute kidney injury populations based on deep clustering: Derivation and interpretation},
                author={Tan, Yongsen and Huang, Jiahui and Zhuang, Jinhu and Huang, Haofan and Tian, Mu and Liu, Yong and Wu, Ming and Yu, Xiaxia},
                journal={International Journal of Medical Informatics},
                volume={191},
                pages={105553},
                year={2024},
                publisher={Elsevier}
            }
    """

    # 1. Index processing (1-based -> 0-based).
    # Although this algorithm does not depend on the values themselves, maintaining consistency is good practice.
    if np.min(BPs) == 1:
        BPs = BPs - 1

    nSmp = BPs.shape[0]
    nTotalBase = BPs.shape[1]

    # Get target number of clusters
    nCluster = get_k_target(n_clusters=nClusters, y=Y)

    labels_list = []
    time_list = []
    M = None

    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        start_idx = iRepeat * nBase
        end_idx = (iRepeat + 1) * nBase

        if start_idx >= nTotalBase:
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        # Slice to get current Ensemble's base partitions
        BPi = BPs[:, start_idx:end_idx]

        t_start = time.time()

        try:
            # --- DCC Consensus Core Logic ---
            # 1. Compute consensus matrix (N x N)
            M = _compute_consensus_matrix(BPi)

            # 2. Run K-Means on the consensus matrix
            kmeans = KMeans(n_clusters=nCluster, random_state=random_seeds[iRepeat], n_init=10)
            label_pred = kmeans.fit_predict(M)

        except Exception as e:
            print(f"DCC Consensus failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros(nSmp)

        t_cost = time.time() - t_start

        labels_list.append(label_pred)
        time_list.append(t_cost)

    if return_matrix:
        return labels_list, time_list, M
    else:
        return labels_list, time_list


def _compute_consensus_matrix(BPs: np.ndarray) -> np.ndarray:
    """
    Compute the consensus matrix (Co-association Matrix).

    Parameters
    ----------
    BPs : np.ndarray
        Base partitions matrix of shape (n_samples, n_base_partitions).

    Returns
    -------
    np.ndarray
        Consensus matrix of shape (n_samples, n_samples).
    """
    n_samples, n_partitions = BPs.shape
    consensus_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    # Optimization: Iterate over base partitions
    # Usually faster than iterating over n_samples compared to original code
    for i in range(n_partitions):
        # Get the i-th base partition result (n_samples,)
        labels = BPs[:, i]

        # Construct connectivity matrix: 1 if in the same cluster, else 0
        # Use broadcasting: (N, 1) == (1, N) -> (N, N)
        # Note: This might cause memory issues if N is very large (>10000); sparse matrix needed for large graphs
        mat = (labels[:, None] == labels[None, :]).astype(np.float32)
        consensus_matrix += mat

    # Normalize
    consensus_matrix /= n_partitions
    return consensus_matrix
