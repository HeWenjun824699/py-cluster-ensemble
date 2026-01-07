============================
Base Clustering Generation
============================

Generating a diverse pool of base partitions (BPs) is the first step in any cluster ensemble experiment. PCE provides multiple strategies to ensure the diversity and quality of these partitions.



Scenario: Generating Bases for a Single Dataset
-----------------------------------------------

If you are focusing on a specific dataset, you can call individual generators from the ``pce.generators`` module. This is useful for testing specific diversity mechanisms like feature perturbation or data subsampling.

.. code-block:: python

   import pce

   # 1. Load your dataset
   data_path = "./data/isolet.mat"
   X, Y = pce.io.load_mat_X_Y(data_path)

   # 2. Generate 200 base partitions using Lite K-Means
   # This method uses different random initializations to create diversity
   BPs = pce.generators.litekmeans(
       X=X,
       Y=Y,
       nClusters=None,  # Automatically determine K range based on Y
       nPartitions=200,
       seed=2026,
       maxiter=100,
       replicates=1
   )

   # 3. Save the pool for subsequent consensus tasks
   pce.io.save_base_mat(BPs, Y, "./data/LKM200/isolet_LKM200.mat")

Scenario: Automated Batch Generation
------------------------------------

For large-scale benchmarking, you may need to generate partitions using multiple algorithms across several datasets. You can define a configuration dictionary to automate this process.

.. code-block:: python

   import os
   import pce

   DATA_DIR = "./data"
   METHODS = {
       "LKM": pce.generators.litekmeans,
       "CDKM": pce.generators.cdkmeans,
       "RSKM": pce.generators.rskmeans,
       "RPKM": pce.generators.rpkmeans,
       "BAGKM": pce.generators.bagkmeans,
       "HETCLU": pce.generators.hetero_clustering
       "SC3": pce.generators.sc3_generator
       "SPECTRAL": pce.generators.spectral
   }

   # Iterate through all datasets
   for filename in [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]:
       X, Y = pce.io.load_mat_X_Y(os.path.join(DATA_DIR, filename))

       for name, func in METHODS.items():
           # Generate and save automatically
           BPs = func(X=X, Y=Y, nPartitions=200, seed=2026)

           out_path = f"./data/{name}200/{filename}_base.mat"
           pce.io.save_base_mat(BPs, Y, out_path)

Supported Generation Strategies
-------------------------------

Depending on your data characteristics, you can choose from the following strategies:

* **Parameter Perturbation**: ``litekmeans`` and ``cdkmeans``.
* **Feature Perturbation**: ``rskmeans`` (subspace sampling) and ``rpkmeans`` (random projection).
* **Data Perturbation**: ``bagkmeans`` (bootstrap resampling).
* **Model Perturbation**: ``hetero_clustering`` (mixing different algorithms like GMM or Spectral Clustering).

For a full list of parameters, please refer to the :doc:`../api/api` section.
