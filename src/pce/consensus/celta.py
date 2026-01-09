import time
from typing import Optional, List

import numpy as np

from .methods.celta_core import celta_core
from .utils.get_k_target import get_k_target


def celta(
        BPs: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        lamb: float = 0.002,
        nBase: int = 20,
        nRepeat: int = 10,
        seed: int = 2025
) -> tuple[list[np.ndarray], list[float]]:
    """
    Clustering Ensemble Meets Low-rank Tensor Approximation (CELTA).

    CELTA models the ensemble problem through a low-rank tensor approximation
    framework. It constructs Microcluster Association (MCA) and Co-Association
    (CA) matrices to capture high-order correlations between samples.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators).
    Y : np.ndarray, optional
        True labels used to infer the number of clusters k.
    nClusters : int, optional
        Target number of clusters k.
    lamb : float, default=0.002
        Regularization parameter for low-rank tensor decomposition.
    nBase : int, default=20
        Number of base partitions used per repetition.
    nRepeat : int, default=10
        Number of experimental repetitions.
    seed : int, default=2025
        Random seed for tensor operations and spectral clustering (standard seed is 2025 for this algorithm).

    Returns
    -------
    labels_list : list of np.ndarray
        List of predicted labels for each repetition.
    time_list : list of float
        List of computation times for each repetition.


    .. note:: **Source**

        Jia et al., "Clustering Ensemble Meets Low-rank Tensor Approximation", *AAAI*, 2021.

        **BibTeX**

        .. code-block:: bibtex

            @inproceedings{jia2021clustering,
                title={Clustering ensemble meets low-rank tensor approximation},
                author={Jia, Yuheng and Liu, Hui and Hou, Junhui and Zhang, Qingfu},
                booktitle={Proceedings of the AAAI conference on artificial intelligence},
                volume={35},
                number={9},
                pages={7970--7978},
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
        # Step B: Run CELTA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        # Explicitly set the global seed to match MATLAB's logic inside the loop
        np.random.seed(current_seed)

        t_start = time.time()

        try:
            # Call core algorithm
            # MATLAB:
            # MCA_ML = compute_MCA_jyh(BPi);
            # CA = compute_CA_jyh(BPi);
            # [A, E, B] = TensorEnsemble(MCA_ML, CA, lambda);
            # W = (A(:, :, 2) + A(:, :, 2)')/2;
            # H_normalized = baseline_SC(W, nCluster);
            # label = litekmeans(H_normalized, nCluster, 'Replicates', 10);

            # Assume Python version core function encapsulates tensor calculation, spectral clustering, and K-Means steps
            label_pred = celta_core(BPi, nCluster, lamb)

            # Ensure output is a flattened numpy array
            label_pred = np.array(label_pred).flatten()

        except Exception as e:
            print(f"CELTA failed on repeat {iRepeat}: {e}")
            # Return all-zero labels on error
            label_pred = np.zeros(nSmp, dtype=int)

        labels_list.append(label_pred)

        t_cost = time.time() - t_start
        time_list.append(t_cost)
        # print(f"Repeat {iRepeat+1}/{nRepeat} finished in {t_cost:.4f}s")

    return labels_list, time_list
