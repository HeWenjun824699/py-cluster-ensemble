import math
from typing import Optional, Union, Tuple, List
import numpy as np


def get_k_range(
        n_smp: int,
        n_clusters: Optional[int] = None,
        y: Optional[np.ndarray] = None
) -> Tuple[int, int]:
    """
    Helper function: Determine the range of cluster number K [min_k, max_k] based on input parameters and data size.

    Logic:
    1. n_clusters is None (Automatic mode):
       - If Y exists (y is not None):
         Use the real number of classes n_real_k from Y.
         Set range to [min(n_real_k, sqrt_n), max(n_real_k, sqrt_n)].
         This leverages prior knowledge while retaining some randomness range.
       - If Y does not exist (y is None):
         Fully unsupervised, default range [2, ceil(sqrt(N))].

    2. n_clusters is int (Fixed mode):
       - User forces a specific K value.
       - min_k = max_k = n_clusters.

    Returns:
        (min_k, max_k) ensuring min_k <= max_k and min_k >= 2
    """
    # Calculate default upper limit: square root of sample number
    sqrt_n = math.ceil(math.sqrt(n_smp))

    # --- 1. Determine initial range ---
    if n_clusters is None:
        if y is None:
            # Case A: Unsupervised automatic range
            # Range: [2, sqrt(N)]
            min_k = 2
            max_k = max(2, sqrt_n)
        else:
            # Case B: Automatic range using Y information
            # Range: [min(real_k, sqrt_n), max(real_k, sqrt_n)]
            n_real_k = len(np.unique(y))
            min_k = min(n_real_k, sqrt_n)
            max_k = max(n_real_k, sqrt_n)

    elif isinstance(n_clusters, int):
        # Case C: User specified fixed K
        min_k = n_clusters
        max_k = n_clusters

    else:
        raise TypeError(f"n_clusters must be None or an integer, got {type(n_clusters)}")

    # --- 2. Safety boundary handling ---
    # Ensure K is at least 2 (prevent calculating 1 or 0)
    min_k = max(2, min_k)
    max_k = max(2, max_k)

    # Ensure min <= max (handle special boundary case where min_k > max_k)
    if min_k > max_k:
        min_k, max_k = max_k, min_k

    return int(min_k), int(max_k)
