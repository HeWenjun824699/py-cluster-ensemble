import os
from .wgraph import wgraph
from .sgraph import sgraph


def metis(x, k):
    # x: similarity matrix
    # k: number of clusters
    
    filename = wgraph(x, None, 0)
    if filename:
        labels = sgraph(k, filename)
        
        # Cleanup graph file
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
                
        return labels
    return None
