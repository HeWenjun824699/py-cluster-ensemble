=====================
Generation Strategies
=====================

The success of a cluster ensemble relies heavily on the **diversity** and **quality** of the base partitions (BPs). PCE provides six distinct strategies to construct diverse clustering pools.


Diversity Mechanisms
--------------------

To create an effective ensemble, PCE utilizes different types of perturbations:

1. **Parameter Perturbation**:

   * **LiteKMeans**: Fast implementation using different random initializations.
   * **CDKMeans**: Utilizes Coordinate Descent to optimize K-means objectives while maintaining diversity.

2. **Feature Perturbation (Subspace)**:

   * **RSKMeans**: Randomly samples feature subsets for each base clustering.
   * **RPKMeans**: Uses random projections (Johnson-Lindenstrauss lemma) to map data into lower-dimensional spaces, preserving distances while changing geometric distribution.

3. **Data Perturbation**:

   * **BagKMeans**: Implements Bootstrap aggregating (Bagging) by subsampling data for training and assigning the rest to the nearest centroids.

4. **Model Perturbation**:

   * **Heterogeneous Clustering**: Combines different algorithms (e.g., Spectral Clustering, GMM, Ward's linkage) to introduce maximum inductive bias diversity.

Choosing the Right Generator
----------------------------

Selecting the appropriate generation strategy is critical for the overall performance of the ensemble:

* **General Purpose & Efficiency**: Use ``LiteKMeans`` for a fast baseline or when dealing with very large datasets.
* **Diversity-Quality Balance**: Use ``CDKMeans`` to obtain higher-quality base partitions while maintaining necessary diversity.
* **High-dimensional Data**: Use ``RSKMeans`` (feature sampling) or ``RPKMeans`` (random projection) to handle the curse of dimensionality.
* **Noisy Data**: Use ``BagKMeans`` to increase the robustness of the ensemble against outliers.
* **Complex Cluster Shapes**: Use ``Hetero-Clustering`` to leverage non-convex algorithms like Spectral Clustering within the ensemble.
