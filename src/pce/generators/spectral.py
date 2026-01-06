from typing import Optional

import numpy as np

from .methods.spectral_core import spectral_core
from .utils.get_k_range import get_k_range


def spectral(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        nPartitions: int = 200,
        seed: int = 2026,
        n_init: int = 100,
        affinity: str = 'nearest_neighbors',
        assign_labels: str = 'discretize'
):
    """
    Spectral Clustering Ensemble Generator.

    Parameters
    ----------
    X : np.ndarray
        Affinity matrix of shape (n_samples, n_samples).
    Y : np.ndarray, optional
        True labels, used to help determine the range of k if nClusters is None.
    nClusters : int, optional
        Fixed number of clusters. If None, a random k is chosen per partition.
    nPartitions : int, default=200
        Number of base partitions to generate (columns in BPs).
    seed : int, default=2026
        Random seed for reproducibility.
    n_init : int, default=100
        Number of times the k-means algorithm will be run with different centroid seeds.
    affinity : str, default='nearest_neighbors'
        How to construct the affinity matrix.
    assign_labels : str, default='discretize'
        Strategy for assigning labels in the embedding space.

    Returns
    -------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, nPartitions).
    """
    nSmp = X.shape[0]

    # Initialize Base Partitions Matrix
    BPs = np.zeros((nSmp, nPartitions), dtype=int)

    # Initialize Random State
    rs = np.random.RandomState(seed)
    # Generate sub-seeds for each partition to ensure reproducibility
    random_seeds = rs.randint(0, 1000001, size=nPartitions)

    for iRepeat in range(nPartitions):
        current_seed = random_seeds[iRepeat]

        # --- Step A: Determine K (Fixed or Random) ---
        if nClusters is not None:
            k = nClusters
        else:
            # Random-k logic (if used as a generic generator)
            min_clusters, max_clusters = get_k_range(n_smp=nSmp, n_clusters=nClusters, y=Y)
            k = np.random.randint(min_clusters, max_clusters + 1)

        # --- Step B: Run Core ---
        labels = spectral_core(
            X=X,
            n_clusters=k,
            seed=current_seed,
            n_init=n_init,
            affinity=affinity,
            assign_labels=assign_labels
        )

        # Store in Matrix
        BPs[:, iRepeat] = labels

    return BPs
