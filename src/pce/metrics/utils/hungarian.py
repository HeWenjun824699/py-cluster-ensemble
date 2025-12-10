import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(A):
    """
    Solve the Assignment problem using the Hungarian method.
    Wrapper around scipy.optimize.linear_sum_assignment to match MATLAB signature roughly.
    
    Parameters:
    A : numpy.ndarray
        Square cost matrix.
        
    Returns:
    C : numpy.ndarray
        The optimal assignment vector. C[j] = i means row i is assigned to column j.
        (Matches MATLAB implementation's output format where index is column, value is row)
    T : float
        The total cost of the optimal assignment.
    """
    A = np.array(A)
    m, n = A.shape
    
    if m != n:
        raise ValueError('Cost matrix must be square!')
    
    # linear_sum_assignment finds min cost
    # returns row_ind, col_ind
    # row_ind is usually [0, 1, ..., n-1]
    # col_ind is the corresponding column for each row
    row_ind, col_ind = linear_sum_assignment(A)
    
    # MATLAB's C vector: C(j) = i means row i assigned to column j.
    # We have row->col mapping. We need col->row mapping.
    C = np.zeros(n, dtype=int)
    for r, c in zip(row_ind, col_ind):
        C[c] = r  # In Python 0-based index. Row r assigned to Col c.
        
    T = A[row_ind, col_ind].sum()
    
    return C, T
