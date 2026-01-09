===========
Quick Start
===========

This guide demonstrates how to use PCE through four common research scenarios.

Scenario A: Automated Batch Pipeline
------------------------------------

The ``pipelines`` module provides a high-level interface to automate the entire "generation-consensus-evaluation" workflow for large-scale comparative experiments.

.. code-block:: python

    from pce.pipelines import consensus_batch

    # Run pipeline
    consensus_batch(
        input_dir='./data',            # Dataset directory
        output_dir='./results',        # Results output directory
        save_format='csv',             # Save format: 'xlsx', 'csv', 'mat'
        consensus_method='cspa',       # Consensus method: 'cspa', 'mcla', 'hgpa' ...
        generator_method='cdkmeans',   # Generator: 'cdkmeans', 'litekmeans' ...
        nPartitions=200,               # Number of base partitions to generate
        seed=2026,                     # Random seed
        nBase=20,                      # Number of base partitions per ensemble
        nRepeat=10,                    # Number of experiment repetitions
        overwrite=True                 # Overwrite existing results
    )

    # Note: Only input_dir is required. See API docs for full parameter list and defaults.

Scenario B: Modular Step-by-Step Call
-------------------------------------

For fine-grained control, you can call each module independently to customize your experimentation workflow.

.. code-block:: python

    import numpy as np
    import pce.io as io
    import pce.generators as gen
    import pce.consensus as con
    import pce.metrics as met

    # 1. Load data (handles .mat format automatically)
    X, Y = io.load_mat_X_Y('./data/isolet.mat')

    # 2. Generate base partitions (using CDK-Means)
    BPs = gen.cdkmeans(X, Y, nPartitions=200)

    # 3. Perform consensus (using CSPA)
    k = len(np.unique(Y))
    labels_list, time_list = con.cspa(BPs, nClusters=k, nBase=20, nRepeat=10)

    # 4. Evaluate results
    results = met.evaluation_batch(labels_list, Y, time_list)

    # 5. Save results to Excel
    io.save_results_xlsx(results, './output/isolet_report.xlsx')

Scenario C: Hyperparameter Grid Search
--------------------------------------

Optimize your ensemble performance by searching across Cartesian product spaces to find the best configuration for your data.

.. code-block:: python

    import pce

    # 1. Prepare paths
    input_dir = './data'
    output_dir = './grid_results'

    # 2. Define grid parameters (param_grid)
    param_grid = {
        'consensus_method': 'cspa',             # Specify consensus method
        't': [20, 50, 100],                     # Explore impact of hyperparameter t
        'k': [5, 10, 15]                        # Explore impact of hyperparameter k
    }

    # 3. Define fixed parameters (fixed_params)
    fixed_params = {
        'generator_method': 'cdkmeans',
        'nPartitions': 200,     # generators
        'seed': 2026,           # generators, consensus
        'maxiter': 100,         # generators
        'replicates': 1,        # generators
        'nBase': 20,            # consensus
        'nRepeat': 10           # consensus
    }

    # 4. Initialize and run
    searcher = pce.grid.GridSearcher(input_dir, output_dir)
    searcher.run(param_grid, fixed_params)

    # Tip: Unsure which parameters an algorithm supports?
    # Use utility function to view parameter list:
    # pce.utils.show_function_params('cspa', module_type='consensus')

Scenario D: Academic Visualization
----------------------------------

PCE features paper-quality visualization tools to analyze results and communicate findings effectively in academic publications.

.. code-block:: python

    import pce.io as io
    import pce.analysis as ana

    # 1. Dimensionality Reduction Scatter Plot (t-SNE/PCA)
    # Use case: Visualize original data distribution / clustering results
    X, Y = io.load_mat_X_Y('./data/isolet.mat')
    ana.plot_2d_scatter(
        X, Y,
        method='tsne',
        title='Ground Truth Visualization (t-SNE)',
        save_path='./output/tsne_plot.png'
    )

    # 2. Co-association Heatmap
    # Use case: Observe ensemble consensus structure (requires BPs from Scenario B)
    BPs, Y = io.load_mat_BPs_Y('./data/base_partitions.mat')
    ana.plot_coassociation_heatmap(
        Y, BPs=BPs,
        title='Ensemble Consensus Matrix',
        save_path='./output/heatmap.png'
    )

    # 3. Performance Metric Trace Plot
    # Use case: Show stability over nRepeat runs from Scenario B
    # results_list is the return value from evaluation_batch in Scenario B
    ana.plot_metric_line(
        results_list,
        metrics=['ACC', 'NMI', 'ARI'],
        xlabel='Run ID',
        title='Performance Stability over 10 Runs',
        save_path='./output/trace_plot.png'
    )

    # 4. Parameter Sensitivity Analysis
    # Use case: Analyze impact of a parameter (e.g., t) on performance from Scenario C
    # csv_file is the summary table generated in Scenario C
    ana.plot_parameter_sensitivity(
        csv_file='./grid_results/isolet_summary.csv',
        target_param='t',    # X-axis: Varying parameter
        metric='ACC',        # Y-axis: Observed metric
        fixed_params={'k': 10}, # Control variables: Fix other parameters
        save_path='./output/sensitivity_t.png'
    )
