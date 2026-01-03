===========
Quick Start
===========

This guide demonstrates how to use PCE through four common research scenarios.

Scenario A: Automated Batch Pipeline
------------------------------------

[cite_start]The ``pipelines`` module provides a high-level interface to automate the entire "generation-consensus-evaluation" workflow for large-scale comparative experiments[cite: 10].

.. code-block:: python

   from pce.pipelines import consensus_batch

   # Automatically scan a directory, generate ensembles, and save results
   consensus_batch(
       input_dir='./data',            # Path to .mat datasets
       output_dir='./results',        # Output directory
       save_format='csv',             # Options: 'csv', 'xlsx', 'mat'
       consensus_method='cspa',       # Chosen consensus algorithm
       nPartitions=200,               # Size of the clustering pool
       nBase=20,                      # Number of base partitions per ensemble
       nRepeat=10                     # Number of experiment repetitions
   )

Scenario B: Modular Step-by-Step Call
-------------------------------------

[cite_start]For fine-grained control, you can call each module independently[cite: 1, 2].

.. code-block:: python

   import pce.io as io
   import pce.generators as gen
   import pce.consensus as con
   import pce.metrics as met

   # 1. Load data (Automatic 1-based to 0-based index correction)
   X, Y = io.load_mat_X_Y('data/sample.mat')

   # 2. Generate Base Partitions
   BPs = gen.cdkmeans(X, Y, nPartitions=200)

   # 3. Execute Consensus
   # Derive final labels using the CSPA algorithm
   labels_list, time_list = con.cspa(BPs, nClusters=10, nBase=20, nRepeat=10)

   # 4. Evaluation
   results = met.evaluation_batch(labels_list, Y, time_list)

Scenario C: Hyperparameter Grid Search
--------------------------------------

[cite_start]Optimize your ensemble performance by searching across Cartesian product spaces[cite: 11].

.. code-block:: python

   from pce.grid import GridSearcher

   # Define the parameter grid
   param_grid = {
       'consensus_method': 'cspa',
       't': [20, 50, 100],  # Hyperparameter t to explore
       'k': [5, 10, 15]     # Hyperparameter k to explore
   }

   searcher = GridSearcher(input_dir='./data', output_dir='./grid_res')
   searcher.run(param_grid, fixed_params={'nBase': 20, 'nRepeat': 10})

Scenario D: Academic Visualization
----------------------------------

[cite_start]PCE features paper-quality visualization tools to analyze results[cite: 9].

.. code-block:: python

   import pce.analysis as ana

   # 1. Plot Dimensionality Reduction (t-SNE/PCA)
   ana.plot_2d_scatter(X, Y, method='tsne', save_path='tsne.pdf')

   # 2. Plot Co-association Heatmap
   ana.plot_coassociation_heatmap(BPs, Y, save_path='heatmap.png')