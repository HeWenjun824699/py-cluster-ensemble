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
       "SC3KM": pce.generators.sc3_kmeans
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

PCE provides **8 distinct strategies** to construct diverse clustering pools, categorized by their underlying diversity mechanisms:

* **K-Means Variants (Speed & Constraints)**:

  * ``litekmeans``: A high-speed implementation using random initializations. Best for massive datasets.
  * ``cdkmeans``: Uses Coordinate Descent with constraints to balance partition quality and diversity.

* **Perturbation Strategies (Data & Feature)**:

  * ``rskmeans``: **Feature Perturbation**. Randomly samples feature subsets (Random Subspace), effective for high-dimensional data.
  * ``rpkmeans``: **Data Projection**. Uses Random Projection (Johnson-Lindenstrauss) to handle ultra-high dimensional or sparse data.
  * ``bagkmeans``: **Data Perturbation**. Uses Bootstrap aggregating (Bagging) to enhance robustness against outliers and noise.

* **Heterogeneous & Domain-Specific**:

  * ``hetero_clustering``: **Model Perturbation**. Combines different inductive biases (e.g., Ward, GMM, Spectral) to capture diverse cluster shapes.
  * ``sc3_kmeans``: **Bio-inspired**. Based on the SC3 algorithm for single-cell data, introducing diversity via random distances and transformations.
  * ``spectral``: **Manifold Learning**. Uses Spectral Clustering to capture non-convex structures that standard K-Means cannot identify.

For a full list of parameters and detailed usage, please refer to the :doc:`../api/api` section.
