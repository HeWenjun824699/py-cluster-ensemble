==========================
Grid Search & Optimization
==========================

In academic research, finding the optimal hyperparameters is crucial for reporting the best possible performance. The ``GridSearcher`` module automates this process by exploring a wide range of parameter combinations to identify the configuration that yields the highest clustering quality.



The searcher calculates the Cartesian product of the defined parameter space and intelligently logs the results for each combination, ensuring your experimental process is both comprehensive and reproducible.

Scenario: Optimizing an Ensemble Algorithm
------------------------------------------

The following example demonstrates how to use ``GridSearcher`` to find the best ``theta`` and ``lamb`` parameters for the ``lwea`` (Locally Weighted Ensemble Algorithm).

Note that the searcher is designed to be "parameter-aware"—it automatically handles the mapping of parameters to the correct modules.

.. code-block:: python

   import pce

   # 1. Define data and output paths
   data_dir = "./data/CDKM200"
   out_dir = "./results/grid/GridSearch_lwea"

   # 2. Define the Grid Parameters (Dynamic Search Space)
   # The searcher will test every combination of these values
   param_grid = {
       'consensus_method': 'lwea',
       'theta': [1, 5, 10, 20],      # Hyperparameter theta for LWEA
       'lamb': [10, 50, 100, 200],   # Hyperparameter lambda for LWEA
   }

   # 3. Define Fixed Parameters (Constant configuration)
   # These can include parameters for both the generator and the consensus method
   fixed_params = {
       'generator_method': 'cdkmeans',
       'nPartitions': 200,     # generator param
       'seed': 2026,           # shared param
       'maxiter': 100,         # generator param
       'replicates': 1,        # generator param
       'nBase': 20,            # consensus param
       'nRepeat': 10           # consensus param
   }

   # 4. Initialize and Execute Grid Search
   # The searcher intelligently filters out irrelevant parameters automatically
   searcher = pce.grid.GridSearcher(data_dir, out_dir)
   searcher.run(param_grid, fixed_params)

Intelligent Parameter Matching
------------------------------

One of the most powerful features of the ``GridSearcher`` is its **internal parameter discovery mechanism**. You do not need to manually separate or filter your configuration dictionary.

* **Automatic Filtering**: The searcher automatically analyzes the function signature of the chosen ``consensus_method``.
* **Redundancy Handling**: It ignores any irrelevant or redundant parameters provided in the ``param_grid`` or ``fixed_params`` (such as skipping generator-specific parameters during the consensus phase).
* **Unified Configuration**: This allows you to maintain a single, unified configuration file even when switching between different ensemble algorithms with varying hyperparameter requirements.

Automated Logging and Results
-----------------------------

Once the search is complete, the ``GridSearcher`` generates several outputs in your designated ``output_dir``:

* **Summary Table**: A CSV file containing the performance metrics (ACC, NMI, ARI, etc.) for every parameter combination.
* **Detailed Logs**: Log files capturing the execution details and any pruning of invalid configurations.
* **JSON Metadata**: A machine-readable file containing the exact search configuration used for the experiment.

.. tip::

   While the searcher filters parameters automatically, you can still use the utility function ``pce.utils.show_function_params('lwea')`` if you want to manually inspect which hyperparameters a specific algorithm is capable of optimizing.
