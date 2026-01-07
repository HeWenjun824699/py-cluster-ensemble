=====================
Generation Strategies
=====================

The success of a cluster ensemble relies heavily on the **diversity** and **quality** of the base partitions (BPs).
PCE currently provides **8 distinct strategies** to construct diverse clustering pools, categorized into three major groups based on their diversity mechanisms.

K-Means Variants (Speed & Constraints)
--------------------------------------

These strategies optimize the classic K-Means objective but introduce specific mechanisms to ensure efficiency or diversity.

* **LiteKMeans**:
  A high-speed implementation of K-Means. It generates diversity primarily through random initializations. It serves as an excellent baseline for very large datasets due to its computational efficiency.

* **CDKMeans (Coordinate Descent K-Means)**:
  Utilizes the Coordinate Descent method to optimize K-Means objectives. It incorporates constraints to balance the quality of individual partitions with the necessary diversity required for the ensemble.

Perturbation Strategies (Data & Feature)
----------------------------------------

These strategies introduce diversity by altering the input data view (features or samples) before clustering.

* **RSKMeans (Random Subspace)**:
  **Feature Perturbation**. Randomly samples subsets of features for each base clustering. This is particularly effective for high-dimensional data by forcing the algorithm to view the data from different feature perspectives.

* **RPKMeans (Random Projection)**:
  **Data Projection**. Uses the Johnson-Lindenstrauss lemma to project data into lower-dimensional spaces via random matrices. It preserves distance structures while significantly changing the geometric distribution, suitable for ultra-high dimensional or sparse data.

* **BagKMeans (Bagging)**:
  **Data Perturbation**. Implements Bootstrap Aggregating (Bagging) by subsampling data points for training and assigning the remaining points to the nearest centroids. This approach enhances robustness against outliers and noise.

Heterogeneous & Domain-Specific
-------------------------------

These strategies introduce diversity at the model level or use domain-specific logic.

* **Hetero-Clustering (Heterogeneous Models)**:
  **Model Perturbation**. Combines fundamentally different algorithms (e.g., Spectral Clustering, Ward's Linkage, GMM, K-Means) within a single pool. This maximizes **inductive bias diversity**, allowing the ensemble to capture various cluster shapes (convex and non-convex) simultaneously.

* **SC3-KMeans**:
  **Bio-inspired Perturbation**. Based on the SC3 algorithm for single-cell RNA-seq data. It introduces diversity through random combinations of distance metrics (Euclidean, Pearson, Spearman), transformation methods (PCA, Laplacian), and feature subspace dimensions.

* **Spectral**:
  Based on Spectral Clustering. It constructs partitions by clustering the eigenvectors of the data's affinity matrix. This is ideal for capturing complex manifold structures or non-convex cluster shapes that K-Means cannot identify.

Choosing the Right Generator
----------------------------

Selecting the appropriate generation strategy is critical for the overall performance of the ensemble:

* **General Purpose & Efficiency**: Use ``LiteKMeans`` for a fast baseline or when dealing with massive datasets.
* **Quality-Diversity Balance**: Use ``CDKMeans`` to obtain higher-quality base partitions while ensuring ensemble diversity.
* **High-dimensional Data**: Use ``RSKMeans`` (feature sampling) or ``RPKMeans`` (random projection) to mitigate the curse of dimensionality.
* **Noisy Data**: Use ``BagKMeans`` to increase the robustness of the ensemble against outliers.
* **Complex/Non-Convex Shapes**: Use ``Hetero-Clustering`` or ``Spectral`` to leverage algorithms capable of detecting non-spherical clusters.
* **Genomics/Bioinformatics**: Use ``SC3-KMeans`` to leverage robust distance and transformation metrics specific to biological data patterns.