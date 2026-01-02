import os
import numpy as np
from .ints import ints


def wgraph(e, w=None, method=0, dataname=None):
    """
    Writes the graph file for METIS.
    Fixed version: Added missing loop for method 0 (CSPA graph).
    """
    if w is None:
        w = []

    # 1. Path handling
    folder = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(folder, 'tmp')

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if dataname is None:
        dataname = os.path.join(tmp_dir, 'graph')

    # Handle filename conflicts (graph0, graph00, ...)
    base_dataname = dataname + str(method)
    dataname = base_dataname
    while os.path.exists(dataname):
        dataname = dataname + str(method)

    # 2. Preprocess matrix
    if method == 0 or method == 1:
        # Remove diagonal elements (self-loops)
        np.fill_diagonal(e, 0)

    # Ensure conversion to integers (METIS can only handle integer weights)
    e = ints(e)
    if method == 1 or method == 3:
        w = ints(w)

    try:
        # Use newline='\n' to ensure consistent line endings
        with open(dataname, 'w', newline='\n') as f:

            # ==========================================
            # Branch A: Standard Graph (Used by CSPA, method=0)
            # ==========================================
            if method == 0 or method == 1:
                # 1. Write Header: Nodes Edges Format
                # Format=1 indicates edge weights
                # Format=11 indicates both vertex and edge weights
                n_edges = int(np.sum(e > 0) / 2)  # Number of edges in undirected graph
                fmt_code = "1" if method == 0 else "11"

                f.write(f"{e.shape[0]} {n_edges} {fmt_code}\n")

                # 2. [Critical Fix] Write specific data
                # Iterate over each node (row)
                for i in range(e.shape[0]):
                    # Find neighbors (positions where e[i, j] > 0)
                    # np.where returns a tuple, take [0]
                    neighbor_indices = np.where(e[i, :] > 0)[0]
                    weights = e[i, neighbor_indices]

                    line_parts = []

                    # If method is 1, also need to write the node's own weight first
                    if method == 1:
                        if w is not None and len(w) > i:
                            line_parts.append(str(int(w[i])))
                        else:
                            line_parts.append("1")  # Default weight

                    # Write neighbor and weight pairs: neighbor weight neighbor weight ...
                    # Note: METIS indexing is 1-based, so idx must be +1
                    for idx, weight in zip(neighbor_indices, weights):
                        line_parts.append(str(idx + 1))  # Neighbor ID (+1)
                        line_parts.append(str(int(weight)))  # Edge weight

                    f.write(" ".join(line_parts) + "\n")

            # ==========================================
            # Branch B: Hypergraph (Used by HGPA/MCLA, method=2/3)
            # ==========================================
            else:
                # Remove empty columns (hyperedges with no nodes)
                col_sums = np.sum(e, axis=0)
                valid_columns = np.where(col_sums > 0)[0]

                if len(valid_columns) != e.shape[1]:
                    e = e[:, valid_columns]
                    if method == 3 and w is not None and len(w) == len(col_sums):
                        w = np.array(w)[valid_columns]

                # Header: NumberOfHyperEdges NumberOfVertices Format(1=EdgeWeights)
                f.write(f"{e.shape[1]} {e.shape[0]} 1\n")

                # Iterate over each hyperedge (column)
                for i in range(e.shape[1]):
                    # Find nodes included in this hyperedge (rows)
                    edges_indices = np.where(e[:, i] > 0)[0]

                    if method == 2:
                        weight = np.sum(e[:, i])
                    else:  # method == 3
                        weight = w[i] if w is not None and len(w) > i else 1

                    line_parts = [str(int(weight))]

                    # Write node indices (1-based)
                    for idx in edges_indices:
                        line_parts.append(str(idx + 1))

                    f.write(" ".join(line_parts) + "\n")

        return dataname

    except IOError as e:
        print(f"wgraph: writing to {dataname} failed: {e}")
        return None
