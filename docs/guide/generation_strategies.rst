=====================
Generation Strategies
=====================

The success of a cluster ensemble relies heavily on the **diversity** and **quality** of the base partitions (BPs). [cite_start]PCE provides six distinct strategies to construct diverse clustering pools[cite: 5].

Diversity Mechanisms
--------------------

To create an effective ensemble, PCE utilizes different types of perturbations:

1. **Parameter Perturbation**:
   * [cite_start]**LiteKMeans**: Fast implementation using different random initializations[cite: 5].
   * [cite_start]**CDK-Means**: Utilizes Coordinate Descent to optimize K-means objectives while maintaining diversity[cite: 5].
2. **Feature Perturbation (Subspace)**:
   * **RSKMeans**: Randomly samples feature subsets for each base clustering. [cite_start]Ideal for high-dimensional data where different features may reveal different structures[cite: 5].
   * [cite_start]**RPKMeans**: Uses random projections (Johnson-Lindenstrauss lemma) to map data into lower-dimensional spaces, preserving distances while changing geometric distribution[cite: 5].
3. **Data Perturbation**:
   * [cite_start]**BagKMeans**: Implements Bootstrap aggregating (Bagging) by subsampling data for training and assigning the rest to the nearest centroids[cite: 5].
4. **Model Perturbation**:
   * [cite_start]**Heterogeneous Clustering**: Combines different algorithms (e.g., Spectral Clustering, GMM, Ward's linkage) to introduce maximum inductive bias diversity[cite: 5].

Choosing the Right Generator
----------------------------

* **High-dimensional data**: Use ``RSKMeans`` or ``RPKMeans``.
* **Noisy data**: Use ``BagKMeans`` for increased robustness.
* **Complex cluster shapes**: Use ``Hetero-Clustering`` to leverage non-convex algorithms.