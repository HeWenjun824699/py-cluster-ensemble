============
Installation
============

This section provides a comprehensive guide to setting up the py-cluster-ensemble (PCE) environment. It covers system requirements, standard installation procedures, and a detailed breakdown of the scientific library stack that powers PCE's core modules. By following these instructions, you will ensure a stable platform for cluster ensemble generation, consensus execution, and academic-grade data analysis.

Requirements
------------

PCE requires **Python 3.10** or higher (Up to 3.12).

Install via pip
---------------

The simplest way to install ``py-cluster-ensemble`` is using ``pip``:

.. code-block:: bash

   pip install py-cluster-ensemble

Dependencies
------------

PCE relies on a robust stack of scientific computing libraries. These will be installed automatically during the pip installation process:

Core Numerical & Machine Learning Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **NumPy (^1.26) & SciPy (^1.11)**: For fundamental numerical operations and matrix manipulations.
* **Scikit-learn (^1.3)**: Provides implementations for base clustering algorithms and dimensionality reduction techniques.
* **Fastcluster (^1.2.6)**: Optimized hierarchical clustering routines used in several consensus methods.
* **Kneed (^0.8.5)**: Used for automatic detection of knee/elbow points in cluster evaluation.

Advanced Consensus & Graph Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **NetworkX (>=3.2)**: Supports fundamental graph-based ensemble algorithms.
* **python-louvain (^0.16)**: Implementation of the Louvain algorithm for community detection.
* **igraph (^0.11) & leidenalg (^0.10)**: High-performance graph manipulation and implementation of the Leiden clustering algorithm.

Deep Learning & Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Torch (^2.2.0)**: Provides the backend for deep clustering and neural network-based computations.
* **Tensorboard (^2.15.0)**: Used for tracking experiments and visualizing training metrics.
* **Tqdm (^4.66)**: Provides interactive progress bars for long-running ensemble algorithms or batch pipelines.

IO & Data Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pandas (^2.2)**: Essential for experiment logging and managing internal data structures.
* **H5py (^3.11)**: Required for ensuring full compatibility with **MATLAB v7.3** ``.mat`` files.
* **Pyreadr (^0.5.4)**: Enables the loading of R-specific ``.rda`` or ``.rds`` files.
* **Xlsxwriter (^3.2)**: Used to generate professionally formatted Excel outputs in the ``pce.io.save_results_xlsx`` module.

Visualization & Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

* **Matplotlib (~3.7.5) & Seaborn (^0.13)**: Power the ``analysis`` module to generate paper-quality 2D scatter plots, co-association heatmaps, and sensitivity charts.

Optional Extras
---------------

For users who wish to build the documentation locally, you can install the optional documentation dependencies:

.. code-block:: bash

   pip install py-cluster-ensemble[docs]

This includes **Sphinx (^7.0)**, **sphinx-rtd-theme (^2.0)**, and **nbsphinx (^0.9)**.
