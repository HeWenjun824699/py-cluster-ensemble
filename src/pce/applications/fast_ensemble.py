import time
import networkx as nx
import csv
import os
from .methods.FastEnsemble.fast_ensemble import fast_ensemble as run_fast_ensemble


def fast_ensemble(
    input_file: str,
    output_file: str,
    n_partitions: int = 10,
    threshold: float = 0.8,
    resolution: float = 0.01,
    algorithm: str = 'leiden-cpm',
    relabel: bool = False,
    delimiter: str = ','
) -> float:
    """
    FastEnsemble wrapper for scalable community detection on large networks.

    Efficiently performs ensemble clustering on graph data by running multiple
    base partitions and utilizing edge pruning with consensus strategies. Designed
    for high scalability on large-scale network edge-lists.

    Parameters
    ----------
    input_file : str
        Path to the input edge-list file (e.g., .txt or .csv) representing
        the graph structure.
    output_file : str
        Path to save the resulting community assignments in CSV format.
    n_partitions : int, default=10
        Number of base partitions to generate in the consensus phase.
    threshold : float, default=0.8
        Weight threshold for edge pruning during consensus graph construction.
    resolution : float, default=0.01
        Resolution parameter for the underlying clustering algorithm (Leiden/Louvain).
    algorithm : str, default='leiden-cpm'
        Choice of community detection algorithm. Supports 'leiden-cpm',
        'leiden-mod', or 'louvain'.
    relabel : bool, default=False
        If True, internally maps node IDs to a continuous range [0, N-1]
        and maps them back in the final output.
    delimiter : str, default=','
        Separator character used in the input edge-list and output CSV file.

    Returns
    -------
    time_cost : float
        The total execution time from reading the graph to saving results.
    """
    
    start_time = time.time()
    
    # 1. Read Input (Edge List)
    # Using nodetype=int to match original implementation's assumption
    try:
        net = nx.read_edgelist(input_file, nodetype=int)
    except Exception as e:
        print(f"Error reading edge list from {input_file}: {e}")
        raise e

    # Optional Relabeling
    reverse_mapping = None
    if relabel:
        mapping = dict(zip(sorted(net), range(0, net.number_of_nodes())))
        net = nx.relabel_nodes(net, mapping)
        reverse_mapping = {y: x for x, y in mapping.items()}

    # 2. Run FastEnsemble Core
    try:
        # Note: run_fast_ensemble expects a NetworkX graph
        partition_dict = run_fast_ensemble(
            net,
            algorithm=algorithm,
            n_p=n_partitions,
            tr=threshold,
            res_value=resolution,
            final_alg=algorithm,  # Simplified: Use same alg for final step
            final_param=resolution,
            weighted='weight' 
        )
    except Exception as e:
        print(f"FastEnsemble execution failed: {e}")
        raise e
    
    # 3. Write Output
    try:
        # Sort keys for consistent output
        keys = list(partition_dict.keys())
        keys.sort()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w', newline='') as out_f:
            writer = csv.writer(out_f, delimiter=delimiter)
            for i in keys:
                cluster_id = partition_dict[i]
                if relabel and reverse_mapping:
                    writer.writerow([reverse_mapping[i], cluster_id])
                else:
                    writer.writerow([i, cluster_id])
                    
    except Exception as e:
        print(f"Error writing output to {output_file}: {e}")
        raise e

    end_time = time.time()
    return end_time - start_time
