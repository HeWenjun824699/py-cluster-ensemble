====================
Consensus Algorithms
====================

As the core engine of the PCE toolkit, the consensus module implements a comprehensive suite of **30+ algorithms** spanning over two decades of research (2003–2025). These methods are designed to derive a single, robust consensus partition from a pool of diverse base clusterings, utilizing diverse theoretical frameworks.

Algorithm Categories
--------------------

PCE categorizes its consensus algorithms into six technical families based on their underlying mechanisms:

1. **The Classics (Foundations)**

   * **CSPA, MCLA, HGPA** (2003): The three foundational pillars of cluster ensembles. They model the problem using similarity matrices, meta-clustering, and hypergraph partitioning, respectively.

2. **Graph Partitioning & Spectral Analysis**

   * **Weighted Graph Methods**: **LWEA** and **LWGP** (2018) introduce local weighting mechanisms to account for the varying reliability of base clusters.
   * **Probability Trajectory**: The **PT-Series** (PTAAL, PTACL, PTASL, PTGP, 2016) uses random walks on co-association graphs to capture the probability trajectories of node transitions.
   * **Spectral & Diffusion**: Includes **USENC** (2020) for ultra-scalable spectral clustering, **CEAM** (2024) for adaptive multiplex diffusion, and iterative methods like **ICSC** (2020) and **ECPCS** (2021).
   * **Self-Enhancement**: **SPCE** (2021) and **ECCMS** (2023) utilize self-paced learning and matrix self-enhancement techniques to improve spectral embedding quality.

3. **Matrix Decomposition & Tensor Learning**

   * These methods capture high-order correlations between samples and clusters beyond pairwise similarities.
   * **Tensor Approximation**: **CELTA** (2021) and **TRCE** (2021) leverage low-rank tensor approximation to robustly learn the consensus matrix.
   * **Graph Tensor**: **GTLEC** (2023) combines graph regularization with tensor learning to optimize the consensus structure.

4. **Optimization & Discrete Methods**

   * **Direct Optimization**: **CDKM** (2022) solves the ensemble problem via Discrete Kernel K-Means with coordinate descent.
   * **Utility Maximization**: **KCC** (2023) transforms consensus clustering into a K-Means problem by maximizing Category Utility or Harmonic Utility functions.
   * **Matrix-Based K-Means**: **DCC** (2024) computes the co-association matrix from base partitions and treats it as similarity features for **K-Means** to derive the final partition.

5. **Diversity & Adaptive Mechanisms**

   * **Multi-Diversity**: The **MDEC-Series** (2022) enhances performance by exploiting structural diversity via Bipartite Graphs, Hierarchical Clustering, or Spectral Clustering.
   * **Adaptive Weighting**: **CDEC** (2025) introduces balance constraints and adaptive weighting to handle varying cluster sizes and quality.

6. **Representation Learning, Active Learning & Domain-Specific**

   * **Representation Learning**: **DREC** (2018) employs dense representation and dual regularization to improve robustness against noise.
   * **Active Learning**: **SPACE** (2024) combines active user queries with self-paced learning to improve results with minimal supervision.
   * **Bioinformatics**: **SC3** (2017) provides a specialized consensus pipeline for Single-Cell RNA-seq data.

Usage Pattern
-------------

PCE is designed with a highly standardized API, ensuring that almost all consensus functions share a consistent calling signature. This uniformity allows researchers to easily swap different algorithms within a single pipeline for performance comparison.

.. code-block:: python

   import pce.consensus as con

   # Standard calling pattern for consensus algorithms
   # labels_list: list of predicted labels for each repetition
   # time_list: list of execution times for each repetition
   labels_list, time_list = con.cspa(
       BPs,             # Base Partitions matrix (N x M)
       nClusters=k,     # Target number of clusters (or None to infer from Y)
       nBase=20,        # Number of base partitions used in each ensemble slice
       nRepeat=10,      # Number of experiment repetitions
       seed=2026        # Random seed for reproducibility
   )

By maintaining this consistent interface, PCE empowers users to conduct rigorous comparative studies across different theoretical frameworks (e.g., comparing a classic graph-based method with a modern active learning approach) with minimal code changes.