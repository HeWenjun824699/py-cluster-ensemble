import os
import torch
import traceback

from .methods.DCC.get_representations import train_representation
from .methods.DCC.consensus_clustering import run_consensus_clustering
from .methods.DCC.visualize_consensus_and_representations import visualize_consensus_and_representations
from .methods.DCC.sensitivity_analysis import run_sensitivity_analysis


def dcc_application(input_path, output_path, input_dim, hidden_dims, k_min=3, k_max=10, run_viz=True, run_sensitivity=True, **kwargs):
    """
    Deep Consensus Clustering Pipeline.

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    output_path : str
        Path to save the output results.
    input_dim : int
        Dimension of the input features.
    hidden_dims : list of int
        List of hidden dimensions for the autoencoder, e.g. [3, 4, 5, ..., 12].
    k_min : int, optional
        Minimum number of clusters, by default 3.
    k_max : int, optional
        Maximum number of clusters, by default 10.
    run_viz : bool, optional
        Whether to run visualization, by default True.
    run_sensitivity : bool, optional
        Whether to run sensitivity analysis (requires labels in data), by default True.
    **kwargs : dict
        Additional keyword arguments passed to internal functions.
    """
    try:
        # Step 1: Representation Learning
        run_representations(input_path, output_path, input_dim, hidden_dims, **kwargs)

        # Step 2: Consensus Clustering
        run_consensus(input_path, output_path, hidden_dims, k_min, k_max)
        
        # Step 3: Analysis & Visualization
        if run_viz or run_sensitivity:
            run_analysis(input_path, output_path, hidden_dims, k_min, k_max, run_viz, run_sensitivity, **kwargs)

        print("\n=== Pipeline Completed Successfully! ===")
        print(f"Results: {os.path.abspath(os.path.join(output_path, 'results'))}")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        traceback.print_exc()


def run_representations(input_path, output_path, input_dim, hidden_dims, **kwargs):
    """
    Run the representation learning step (Step 1).

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    output_path : str
        Path to save the output results.
    input_dim : int
        Dimension of the input features.
    hidden_dims : list of int
        List of hidden dimensions to iterate over.
    **kwargs : dict
        Additional arguments for the training function (e.g. epochs, lr).
    """
    print("\n=== Step 1: Generating Representations ===")

    use_cuda = 1 if torch.cuda.is_available() else 0
    print(f"CUDA Available: {torch.cuda.is_available()} (use_cuda={use_cuda})")

    # Loop to generate representations for different hidden dimensions
    for h_dim in hidden_dims:
        print(f"\n--> [Pipeline] Processing hidden_dims={h_dim}...")

        train_representation(
            input_path=input_path,
            output_path=output_path,
            input_dim=input_dim,
            hidden_dim=h_dim,
            cuda=use_cuda,
            **kwargs
        )


def run_consensus(input_path, output_path, hidden_dims, k_min=3, k_max=10, **kwargs):
    """
    Run the consensus clustering step (Step 2).

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    output_path : str
        Path to save the output results.
    hidden_dims : list of int
        List of hidden dimensions used in the previous step.
    k_min : int, optional
        Minimum number of clusters, by default 3.
    k_max : int, optional
        Maximum number of clusters, by default 10.
    **kwargs : dict
        Additional arguments.
    """
    print("\n=== Step 2: Consensus Clustering ===")

    # Call the refactored consensus clustering function directly
    run_consensus_clustering(
        input_path=input_path,
        output_path=output_path,
        hidden_dims=hidden_dims,
        k_min=k_min,
        k_max=k_max,
        ** kwargs
    )


def run_analysis(input_path, output_path, hidden_dims, k_min, k_max, run_viz=True, run_sensitivity=True, **kwargs):
    """
    Run the analysis and visualization step (Step 3).

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    output_path : str
        Path to save the output results.
    hidden_dims : list of int
        List of hidden dimensions.
    k_min : int
        Minimum number of clusters.
    k_max : int
        Maximum number of clusters.
    run_viz : bool, optional
        Whether to generate visualizations, by default True.
    run_sensitivity : bool, optional
        Whether to run sensitivity analysis, by default True.
    **kwargs : dict
        Additional arguments, e.g. 'n_bootstrapping', 'bootstrapping_proportion'.
    """
    print("\n=== Step 3: Analysis & Visualization ===")
    
    results_dir = os.path.join(output_path, 'results/pkls')
    representations_dir = os.path.join(output_path, 'representations')
    analysis_dir = os.path.join(output_path, 'analysis')
    
    # Iterate through k values (though original scripts often focused on a specific k, we can loop or pick one)
    # For now, let's process all k in range if files exist
    
    for k in range(k_min, k_max + 1):
        if not os.path.exists(os.path.join(results_dir, f'consensus_cluster_{k}.pkl')):
            continue
            
        print(f"\n--> Processing k={k}...")
        
        if run_viz:
            visualize_consensus_and_representations(
                results_dir=results_dir,
                representations_dir=representations_dir,
                output_dir=os.path.join(analysis_dir, f'k_{k}'),
                k=k,
                hidden_dims=hidden_dims
            )
            
        if run_sensitivity:
            # sensitivity analysis is computationally expensive, maybe optional or lower N
            run_sensitivity_analysis(
                data_path=input_path,
                results_dir=results_dir,
                k=k,
                output_dir=os.path.join(analysis_dir, f'k_{k}'),
                n_bootstrapping=kwargs.get('n_bootstrapping', 100),  # Default lower for speed in pipeline
                bootstrapping_proportion=kwargs.get('bootstrapping_proportion', 0.7)
            )
