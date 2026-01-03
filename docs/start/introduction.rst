============
Introduction
============

**py-cluster-ensemble (PCE)** is a comprehensive Python toolkit designed for cluster ensemble generation, consensus algorithms, and automated experimentation. It specifically serves researchers and the academic community by providing a unified framework for complex ensemble clustering tasks.

PCE covers the complete technical workflow, from constructing base clustering pools to deriving final ensemble consensus partitions.

Project Metadata
----------------

* **Version**: 1.0.0
* **License**: MIT
* **Repository**: https://github.com/HeWenjun824699/py-cluster-ensemble
* **Target Audience**: Researchers, Data Scientists, and Academic Community.

Core Capabilities
-----------------

PCE fills the gap in the Python ecosystem for a unified ensemble clustering tool by offering:

Extensive Algorithm Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements a wide range of **27+ consensus algorithms** spanning over two decades of research, covering the period from **2003 to 2025**. This includes classic graph-based methods (CSPA, MCLA), matrix/tensor decomposition, and the latest adaptive weighting strategies.

Seamless MATLAB Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides a robust data interface for interacting with MATLAB ``.mat`` files (including **v7.3/HDF5** compatibility). It features automatic index correction to handle the transition from MATLAB's 1-based indexing to Python's 0-based indexing effortlessly.

Academic Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~

Features automated experimentation pipelines and an intelligent **GridSearcher** for hyperparameter optimization. These tools ensure that large-scale experiments are consistent, reproducible, and easy to log.

Paper-Quality Analysis
~~~~~~~~~~~~~~~~~~~~~~

Includes built-in visualization tools designed for academic publication. Users can generate high-quality **t-SNE/PCA** reduction plots, **co-association heatmaps**, and **parameter sensitivity** charts with support for vector graphics (PDF) output.

Comprehensive Metrics
~~~~~~~~~~~~~~~~~~~~~

Offers a suite of **14 clustering validation metrics**, such as ACC, NMI, and ARI. It supports both single-run verification and batch statistical analysis (calculating mean and standard deviation) to provide rigorous evaluation of clustering results.
