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
    FastEnsemble Application Entry Point.
    
    Reads a network edge-list from a file, performs ensemble clustering, 
    and writes the community assignments to an output file.

    Parameters
    ----------
    input_file : str
        Path to the input edge-list file.
    output_file : str
        Path where the output community file (CSV) will be saved.
    n_partitions : int, default=10
        Number of partitions in consensus clustering.
    threshold : float, default=0.8
        Threshold value for consensus edge pruning.
    resolution : float, default=0.01
        Resolution parameter for the Leiden algorithm.
    algorithm : str, default='leiden-cpm'
        Clustering algorithm ('leiden-cpm', 'leiden-mod', or 'louvain').
    relabel : bool, default=False
        Whether to relabel network nodes from 0 to #nodes-1.
    delimiter : str, default=' '
        Delimiter used in the output CSV file.

    Returns
    -------
    time_cost : float
        Execution time in seconds.
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
