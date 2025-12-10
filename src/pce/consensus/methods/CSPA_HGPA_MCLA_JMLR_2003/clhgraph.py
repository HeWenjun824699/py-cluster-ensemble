from .hmetis import hmetis


def clhgraph(x, k):
    """
    Provides cluster labels 1 to k from hypergraph partitioning.
    sfct is ignored to match MATLAB implementation.
    
    Args:
        x: Hypergraph incidence matrix
        k: Number of clusters
        
    Returns:
        cl: Cluster labels
    """
    # STRICT MATCH WITH MATLAB: sfct is IGNORED.
    # MATLAB: cl = hmetis(x,k);
    cl = hmetis(x, k)
    return cl
