import numpy as np

def prox_l1(b, lambd):
    """
    The proximal operator of the l1 norm
    min_x lambda*||x||_1+0.5*||x-b||_2^2
    """
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)
