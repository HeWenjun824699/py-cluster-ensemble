==========================
Grid Search & Optimization
==========================

In academic research, finding the optimal hyperparameters is crucial for reporting the best possible performance. [cite_start]The ``GridSearcher`` module automates this process[cite: 11].



[cite_start]The searcher calculates the Cartesian product of the parameter space and intelligently logs the results for each combination[cite: 11].

.. code-block:: python

   import pce

   input_dir = './data'
   output_dir = './grid_results'

   # Define dynamic search grid
   param_grid = {
       'consensus_method': 'cspa',
       't': [20, 50, 100],  # Exploring parameter 't'
       'k': [5, 10, 15]     # Exploring target cluster count
   }

   # Define static experimental settings
   fixed_params = {
       'generator_method': 'cdkmeans',
       'nPartitions': 200,
       'seed': 2026,
       'nBase': 20,
       'nRepeat': 10
   }

   # Initialize and run
   searcher = pce.grid.GridSearcher(input_dir, output_dir)
   searcher.run(param_grid, fixed_params)

[cite_start]**Tip**: If you are unsure which parameters a specific algorithm supports, use the utility function ``pce.utils.show_function_params('cspa')``[cite: 12].