====================
Consensus Algorithms
====================

[cite_start]As the core engine of PCE, the consensus module implements a comprehensive range of algorithms (2003–2025) to derive final partitions from base clustering matrices[cite: 6].

Algorithm Categories
--------------------

PCE categorizes its 27+ consensus algorithms into several technical families:

1. **Graph-Based Methods**:
   * [cite_start]Classic algorithms like **CSPA**, **MCLA**, and **HGPA** (2003) which model the problem as graph or hypergraph partitioning[cite: 6].
   * [cite_start]Modern weighted graph methods like **LWEA** and **LWGP** (2018)[cite: 6].
2. **Matrix & Tensor-Based Methods**:
   * [cite_start]Methods that treat ensembles as low-rank approximation or tensor decomposition problems, such as **CELTA**, **GTLEC**, and **TRCE** (2021/2023)[cite: 6].
3. **Optimized Linkage Methods**:
   * [cite_start]Probability trajectory-based methods like **PTAAL**, **PTACL**, and **PTASL** (2016)[cite: 6].
4. **Advanced & Hybrid Strategies**:
   * [cite_start]**SPACE (2024)**: Combines Active Learning with Self-Paced Learning for semi-supervised refinement[cite: 7].
   * [cite_start]**CDEC (2025)**: Introduces balance-adaptive weighting to handle cluster size constraints[cite: 7].

Usage Pattern
-------------

Most consensus functions in PCE follow a standardized API:

.. code-block:: python

   labels_list, time_list = pce.consensus.method_name(BPs, nClusters=k, nBase=20, nRepeat=10)

[cite_start]This design allows researchers to easily swap algorithms in a single pipeline to compare performance across different theoretical frameworks[cite: 6, 10].