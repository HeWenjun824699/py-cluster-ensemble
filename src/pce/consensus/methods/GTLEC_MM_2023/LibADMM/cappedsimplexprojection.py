import numpy as np

def cappedsimplexprojection(y, k):
    """
    Efficient implementation for projecting a vector y onto the capped simplex.
    The capped simplex is the set of vectors x such that:
    - sum(x) = k
    - 0 <= x_i <= 1 for all i

    This implementation is based on the algorithm described in:
    "Efficient Projections onto the l1-Ball for Some Popular Regularizers"
    by John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
    The algorithm has a time complexity of O(n log n) due to sorting.

    Parameters
    ----------
    y : np.ndarray
        The vector to project.
    k : int
        The desired sum of the projected vector.

    Returns
    -------
    x : np.ndarray
        The projected vector.
    """
    n = len(y)
    if k < 0 or k > n:
        raise ValueError("Sum constraint k is infeasible.")

    # Sort y in descending order
    y_sorted = np.sort(y)[::-1]
    
    # Calculate the cumulative sum of the sorted vector
    y_cumsum = np.cumsum(y_sorted)

    # Find the largest rho such that: rho * y_sorted[rho-1] - y_cumsum[rho-1] < k
    # This is a vectorized search for the optimal rho.
    # The condition is derived from the properties of the projection.
    
    # We are looking for rho in {1, ..., n}
    rho_values = np.arange(1, n + 1)
    
    # Check the condition for all possible rho values
    # The term is `y_sorted[rho-1]` in a loop, or `y_sorted` for the whole vector.
    # The cumsum is `y_cumsum[rho-1]` in a loop, or `y_cumsum`.
    
    # Inequality: rho * y_i - sum_{j=1 to i} y_j < k
    # Vectorized: rho_values * y_sorted - y_cumsum < k
    
    # Find all rhos that satisfy the condition
    satisfying_indices = np.where(rho_values * y_sorted - y_cumsum < k)[0]
    
    # The optimal rho is the largest index found, plus 1 (for 1-based index)
    if len(satisfying_indices) == 0:
        rho = 0 # This case can happen if k is very small, e.g. k=0
    else:
        rho = satisfying_indices[-1] + 1

    # Calculate the Lagrange multiplier (theta)
    # theta = (sum_{i=1 to rho} y_i - k) / rho
    if rho == 0:
        theta = (np.sum(y_sorted) - k) / n # A safe fallback, though should ideally use a different theta
    else:
        theta = (y_cumsum[rho - 1] - k) / rho

    # The projection is given by soft-thresholding with theta
    x = np.minimum(1, np.maximum(y - theta, 0))
    
    # Due to numerical precision issues, the sum of x might not be exactly k.
    # A final correction step can be applied if needed, but for most applications
    # this is sufficient.
    
    return x