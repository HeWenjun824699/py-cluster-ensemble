import numpy as np
from .checks import checks
from .cmetis import cmetis
from .simbjac import simbjac


def clcgraph(x, k, sfct='simbjac'):
    """
    Cluster Graph Partitioning.
    
    MATLAB equivalent:
    function cl = clcgraph(x,k,sfct)
    """
    
    # Resolve similarity function
    if isinstance(sfct, str):
        if sfct == 'simbjac':
            sim_func = simbjac
        else:
             raise ValueError(f"Unknown similarity function string: {sfct}")
    elif callable(sfct):
        sim_func = sfct
    else:
        raise TypeError("sfct must be a string or callable")

    # Compute similarity
    similarity_matrix = sim_func(x)

    # checks(feval(sfct,x))
    # Assuming checks.checks validates/fixes the matrix
    checked_matrix = checks(similarity_matrix)
    
    # sum(x, 2)
    # Calculate weights based on input x
    # If x is sparse, sum returns a matrix object, so we flatten it
    # If x is dense, sum returns an array
    if hasattr(x, 'sum'): # numpy array or sparse matrix
        weights = x.sum(axis=1)
        if hasattr(weights, 'A'): # matrix object (sparse sum result)
            weights = weights.A.flatten()
    else:
        weights = np.sum(x, axis=1)

    # cl = cmetis(..., weights, k)
    cl = cmetis(checked_matrix, weights, k)
    
    return cl
