from typing import Optional
import numpy as np


def get_k_target(
        n_clusters: Optional[int] = None,
        y: Optional[np.ndarray] = None
) -> int:
    """
    Helper function: Determine the target number of clusters K (k_target) for ensemble algorithms.

    Used for ensemble algorithms such as CSPA, MCLA, HGPA.

    Logic priority:
    1. n_clusters (int): Explicitly specified by the user, highest priority.
    2. Y (array): User did not specify K, but provided Y. Infer K = len(unique(Y)).
    3. Error: Neither n_clusters nor Y is provided, raise ValueError.

    Parameters:
        n_clusters: Number of clusters input by user (must be int or None).
        y: True labels (used to infer K).

    Returns:
        k_target (int)
    """

    # Priority 1: User explicitly specified (Fixed K)
    if n_clusters is not None:
        # [Critical Protection] Prevent user from passing float (3.0), str ("3"), or tuple
        if not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be an integer, got {type(n_clusters)}")
        # [Extra Protection] Prevent user from passing negative numbers or 0
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

        return n_clusters

    # Priority 2: User did not specify nClusters, but provided Y (compatible with test mode)
    elif y is not None:
        return len(np.unique(y))

    # Priority 3: Neither K nor Y provided -> Error
    else:
        raise ValueError("n_clusters must be provided if Y is None (Unsupervised mode).")
