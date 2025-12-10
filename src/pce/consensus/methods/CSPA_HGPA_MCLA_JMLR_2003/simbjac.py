from .simxjac import simxjac


def simbjac(a, b=None):
    """
    Computes Jaccard similarity between row objects in matrices a and b.
    
    MATLAB equivalent:
    function s = simbjac(a,b)
    """
    if b is None:
        b = a
    
    # s = simxjac(a>0,b>0);
    return simxjac((a > 0).astype(float), (b > 0).astype(float))
