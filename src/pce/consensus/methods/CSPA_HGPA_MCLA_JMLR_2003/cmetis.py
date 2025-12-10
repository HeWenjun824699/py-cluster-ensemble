import os
from .wgraph import wgraph
from .sgraph import sgraph


def cmetis(e, w, k):
    """
    METIS partitioning on weighted graph.
    
    MATLAB equivalent:
    function labels=cmetis(e,w,k) 
    """
    # filename = wgraph(e,w,1);
    # We assume wgraph.wgraph writes the graph to a file and returns the filename
    filename = wgraph(e, w, 1)
    
    # labels = sgraph(k,filename);
    # We assume sgraph.sgraph runs the partitioner and reads the result
    labels = sgraph(k, filename)
    
    # delete(filename);
    if filename and os.path.exists(filename):
        try:
            os.remove(filename)
        except OSError:
            pass
        
    return labels
