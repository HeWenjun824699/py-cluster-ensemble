====================
Consensus Algorithms
====================

As the core engine of the PCE toolkit, the consensus module implements a comprehensive suite of algorithms spanning over two decades of research (2003–2025). These methods are designed to derive a single, robust consensus partition from a pool of diverse base clusterings.

Algorithm Categories
--------------------

PCE categorizes its 30+ consensus algorithms into several technical families based on their underlying theoretical frameworks:

1. **Graph-Based Methods**:

   * **Classic Graph Partitioning**: Includes foundational algorithms like **CSPA**, **MCLA**, and **HGPA** (2003). These model the ensemble problem as a graph or hypergraph partitioning task to find the optimal cut.
   * **Weighted Graph Methods**: Modern extensions such as **LWEA** and **LWGP** (2018), which introduce local weighting mechanisms to account for the varying reliability of base clusters.

2. **Matrix & Tensor-Based Methods**:

   * These methods treat the ensemble task as a optimization problem. Examples include **CELTA** (2021), **GTLEC** (2023), and **TRCE** (2021), which leverage low-rank matrix approximation or tensor decomposition to capture high-order correlations between samples and clusters.

3. **Optimized Linkage & Probability Trajectories**:

   * **Probability Trajectory (PT) Methods**: Includes **PTAAL**, **PTACL**, **PTASL**, and **PTGP** (2016). These algorithms use random walks on a co-association graph to define "probability trajectories," which are then used to optimize linkage criteria.

4. **Advanced & Hybrid Strategies**:

   * **Active & Self-Paced Learning**: **SPACE** (2024) integrates active learning to query informative constraints, combined with self-Paced learning to gradually incorporate them into the consensus process.
   * **Adaptive Weighting**: **CDEC** (2025) introduces a balance-adaptive weighting mechanism to handle cluster size constraints and adaptive importance for different base partitions.

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
       nClusters=k,     # Target number of clusters
       nBase=20,        # Number of base partitions used in each ensemble
       nRepeat=10,      # Number of experiment repetitions
       seed=2026        # Random seed for reproducibility
   )

By maintaining this consistent interface, PCE empowers users to conduct rigorous comparative studies across different theoretical frameworks (e.g., comparing a graph-based method with a tensor-based method) with minimal code changes.