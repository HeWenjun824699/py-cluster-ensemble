from .wgraph import wgraph
from .sgraph import sgraph
import os


def hmetis(x, k, w=None):
    """
    Performs hypergraph partitioning using hMETIS.
    
    Args:
        x: Hypergraph incidence matrix (Vertices x Hyperedges)
        k: Number of partitions
        w: Weights for hyperedges (optional)
        
    Returns:
        labels: Array of cluster labels
    """
    if w is None:
        filename = wgraph(x, None, 2)
    else:
        filename = wgraph(x, w, 3)
    
    if filename is None:
        return None

    labels = sgraph(k, filename)
    
    # Clean up the graph file created by wgraph
    try:
        os.remove(filename)
    except OSError:
        pass
        
    return labels
