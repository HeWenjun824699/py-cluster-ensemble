=============================
Automated Batch Processing
=============================

This tutorial covers how to handle multiple datasets efficiently using PCE's high-level pipelines.

Automated Pipeline (Recommended)
--------------------------------

[cite_start]The ``consensus_batch`` pipeline is designed for large-scale experiments[cite: 10]. [cite_start]It automatically scans a directory for ``.mat`` files, generates base partitions if they are missing, runs the ensemble algorithm, and exports results[cite: 10].

.. code-block:: python

   from pce.pipelines import consensus_batch

   # Run the automated pipeline
   consensus_batch(
       input_dir='./data',            # Directory containing .mat datasets
       output_dir='./results',        # Directory to save outputs
       save_format='csv',             # Options: 'xlsx', 'csv', 'mat'
       consensus_method='cspa',       # Ensemble algorithm
       generator_method='cdkmeans',   # Generator if BPs are missing
       nPartitions=200,               # Total pool size
       nBase=20,                      # Base partitions per ensemble
       nRepeat=10                     # Number of repetitions
   )

Modular Step-by-Step Workflow
-----------------------------

[cite_start]For researchers who need to intervene at specific stages (e.g., custom data preprocessing), PCE allows manual modular calls[cite: 1, 2].

1. [cite_start]**Loading Data**: PCE handles MATLAB v7.3 compatibility and 1-based index correction automatically[cite: 3, 4].
2. [cite_start]**Generating Pools**: Create a diverse pool of clusterings using strategies like Coordinate Descent K-Means[cite: 5].
3. [cite_start]**Running Consensus**: Derive final labels using algorithms like CSPA or MCLA[cite: 6].
4. [cite_start]**Evaluating**: Calculate 14 different metrics including ACC and NMI[cite: 7, 8].

.. code-block:: python

   import pce.io as io
   import pce.generators as gen
   import pce.consensus as con
   import pce.metrics as met

   # Load dataset
   X, Y = io.load_mat_X_Y('data/isolet.mat')

   # Generate 200 base partitions
   BPs = gen.cdkmeans(X, Y, nPartitions=200)

   # Run ensemble (e.g., 10 repeats with 20 base partitions each)
   k = len(np.unique(Y))
   labels_list, time_list = con.cspa(BPs, nClusters=k, nBase=20, nRepeat=10)

   # Evaluate and save
   results = met.evaluation_batch(labels_list, Y, time_list)
   io.save_results_xlsx(results, 'output/isolet_report.xlsx')