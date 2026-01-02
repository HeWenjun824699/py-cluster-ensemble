from typing import Optional, List, Union
import numpy as np

from .methods.hetero_clustering_core import hetero_clustering_core
from .utils.check_array import check_array
from .utils.get_k_range import get_k_range


def hetero_clustering(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        algorithms: Union[str, List[str]] = 'auto',
        seed: int = 2026
):
    """
    Heterogeneous Ensemble Generator

    Principle: Mix clustering algorithms with different Inductive Biases.
    Combines:
    1. Model Perturbation: Randomly select algorithm (Spectral, Hierarchical, GMM)
    2. Parameter Perturbation: Random K value (Random-k)

    Parameters
    ----------
    algorithms : str or List[str]
        - 'auto': Randomly mix ['spectral', 'ward', 'average', 'complete', 'gmm', 'kmeans']
        - List[str]: Specify a subset, e.g. ['spectral', 'ward']
        - str: Fix one algorithm, e.g. 'spectral'
    """
    nSmp = X.shape[0]

    # [Core Modification] Automatically handle all format issues
    X = check_array(X, accept_sparse=False)

    # --- 1. Configure algorithm pool ---
    if algorithms == 'auto':
        # Default mix strategy (added 'complete')
        algo_pool = ['spectral', 'ward', 'average', 'complete', 'gmm', 'kmeans']
    elif isinstance(algorithms, str):
        algo_pool = [algorithms]
    else:
        algo_pool = algorithms

    # --- 2. Call helper function to get K value range ---
    minCluster, maxCluster = get_k_range(n_smp=nSmp, n_clusters=nClusters, y=Y)

    # --- 3. Generate base partitions ---
    BPs = np.zeros((nSmp, nPartitions), dtype=np.float64)

    nRepeat = nPartitions

    # Initialize random number generator
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    # Pre-generate algorithm selection sequence (uniform distribution)
    selected_algos = rs.choice(algo_pool, size=nRepeat)

    for iRepeat in range(nRepeat):
        current_seed = random_seeds[iRepeat]
        current_algo = selected_algos[iRepeat]

        # -------------------------------------------------
        # Step A: Randomly select K value (Random-k)
        # -------------------------------------------------
        np.random.seed(current_seed)

        if minCluster == maxCluster:
            iCluster = minCluster
        else:
            iCluster = np.random.randint(minCluster, maxCluster + 1)

        # -------------------------------------------------
        # Step B: Run selected heterogeneous algorithm
        # -------------------------------------------------
        try:
            label = hetero_clustering_core(
                X=X,
                n_clusters=iCluster,
                algorithm=current_algo,
                seed=current_seed
            )

            # Store results (convert to 1-based)
            BPs[:, iRepeat] = label + 1

        except Exception as e:
            # Fault tolerance: Some algorithms (e.g., Spectral) may fail on specific data
            # If failed, fallback to K-Means to ensure process continuity
            # print(f"Warning: {current_algo} failed at iter {iRepeat}. Fallback to kmeans. Error: {e}")
            fallback_label = hetero_clustering_core(X, iCluster, 'kmeans', seed=current_seed)
            BPs[:, iRepeat] = fallback_label + 1

    return BPs
