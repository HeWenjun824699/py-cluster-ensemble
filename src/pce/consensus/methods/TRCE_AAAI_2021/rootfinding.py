import numpy as np

def rootfinding(x, c, h):
    """
    Root finding function corresponding to rootfinding.m
    
    y = sum(c ./ (h + x)) - 1
    
    Args:
        x: scalar
        c: vector
        h: vector
    
    Returns:
        y: scalar
    """
    return np.sum(c / (h + x)) - 1
