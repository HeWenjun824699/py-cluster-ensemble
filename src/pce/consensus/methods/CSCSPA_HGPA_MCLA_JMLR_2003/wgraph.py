import os
import numpy as np
from .ints import ints

def wgraph(e, w=None, method=0, dataname=None):
    if w is None:
        w = []
    
    folder = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(folder, 'tmp')
    
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        
    if dataname is None:
        dataname = os.path.join(tmp_dir, 'graph')
        
    base_dataname = dataname + str(method)
    dataname = base_dataname
    
    # Handle filename collision
    while os.path.exists(dataname):
        dataname = dataname + str(method)
        
    if method == 0 or method == 1:
        # Zero out diagonal
        np.fill_diagonal(e, 0)
        
    e = ints(e)
    if method == 1 or method == 3:
        w = ints(w)
        
    try:
        with open(dataname, 'w') as f:
            if method == 0:
                # Format: Nodes Edges 1 (Weighted)
                n_edges = int(np.sum(e > 0) / 2)
                f.write(f"{e.shape[0]} {n_edges} 1\n")
            elif method == 1:
                n_edges = int(np.sum(e > 0) / 2)
                f.write(f"{e.shape[0]} {n_edges} 11\n")
            else:
                # Hypergraph (simplified handling)
                f.write(f"{e.shape[1]} {e.shape[0]} 1\n")

            if method == 0:
                for i in range(e.shape[0]):
                    # indices: 0-based in numpy, convert to 1-based for METIS
                    edges_indices = np.where(e[i, :] > 0)[0]
                    weights = e[i, edges_indices]
                    
                    line_parts = []
                    for idx, weight in zip(edges_indices, weights):
                        line_parts.append(str(idx + 1))
                        line_parts.append(str(int(weight)))
                    
                    f.write(" ".join(line_parts) + "\n")
            elif method == 1:
                # Method 1 not fully implemented in this port as CSPA uses method 0
                pass
            else:
                 # Method 2/3 not fully implemented
                pass

        return dataname
    except IOError as e:
        print(f"wgraph: writing to {dataname} failed: {e}")
        return None
