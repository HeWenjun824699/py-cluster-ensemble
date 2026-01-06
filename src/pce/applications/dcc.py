import os
import torch
import traceback

from .methods.Deep_Consensus_Clustering.get_representations import train_representation
from .methods.Deep_Consensus_Clustering.consensus_clustering import run_consensus_clustering
from .methods.Deep_Consensus_Clustering.visualize_consensus_and_representations import visualize_consensus_and_representations
from .methods.Deep_Consensus_Clustering.sensitivity_analysis import run_sensitivity_analysis


def dcc_application(input_path, output_path, input_dim, hidden_dims, k_min=3, k_max=10, run_viz=True, run_sensitivity=True, **kwargs):
    """
    Deep Consensus Clustering Pipeline
    args:
        hidden_dims: list of integers, e.g. [3, 4, 5, ..., 12]
        run_viz: Whether to run visualization.
        run_sensitivity: Whether to run sensitivity analysis (requires labels in data).
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
    print("\n=== Step 1: Generating Representations ===")

    use_cuda = 1 if torch.cuda.is_available() else 0
    print(f"CUDA Available: {torch.cuda.is_available()} (use_cuda={use_cuda})")

    # 循环调用函数生成不同维度的特征
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
    print("\n=== Step 2: Consensus Clustering ===")

    # 直接调用重构后的函数
    run_consensus_clustering(
        input_path=input_path,
        output_path=output_path,
        hidden_dims=hidden_dims,
        k_min=k_min,
        k_max=k_max,
        ** kwargs
    )


def run_analysis(input_path, output_path, hidden_dims, k_min, k_max, run_viz=True, run_sensitivity=True, **kwargs):
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
                output_dir=os.path.join(analysis_dir, 'analysis', f'k_{k}'),
                k=k,
                hidden_dims=hidden_dims
            )
            
        if run_sensitivity:
            # sensitivity analysis is computationally expensive, maybe optional or lower N
            run_sensitivity_analysis(
                data_path=input_path,
                results_dir=results_dir,
                k=k,
                output_dir=os.path.join(analysis_dir, 'analysis', f'k_{k}'),
                n_bootstrapping=kwargs.get('n_bootstrapping', 100),  # Default lower for speed in pipeline
                bootstrapping_proportion=kwargs.get('bootstrapping_proportion', 0.7)
            )
