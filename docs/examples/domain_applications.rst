===================
Domain Applications
===================

The ``applications`` module encapsulates end-to-end solutions for specific research fields.

Single-Cell RNA-Seq (SC3)
-------------------------

PCE includes a Python implementation of the SC3 algorithm for single-cell consensus clustering. It integrates gene filtering and biological analysis.

.. code-block:: python

    from pce.io.load_rda import load_rda_X_Y
    from pce.applications import sc3_application

    # 1. Load data
    X, Y, gene_names, cell_names = load_rda_X_Y("./data/yan.rda")
    
    # 2.Run SC3 on expression data
    labels, biology_res, time_cost = sc3_application(
        X=X,
        Y=None,
        nClusters=None,
        gene_names=gene_names,
        cell_names=cell_names,
        output_directory="./sc3_output",
        biology=True,
        gene_filter=True,
        seed=2026
    )


Brain Network Analysis (ICSC)
-----------------------------

The ICSC application is designed for consensus clustering on brain network data (e.g., fMRI). It supports two primary modes: Group-Level (Multiple Subjects) and Individual-Level (Subject Sessions) analysis.

**Multiple Subjects Mode:**

Suitable for identifying common functional networks across a population.

.. code-block:: python

    import pce

    pce.applications.icsc_mul_application(
        dataset='multiple_subjects',
        data_directory='./data/multiple_subjects',
        save_dir='./outputs/multiple_subjects',
        num_nodes=264,
        num_threads=1,
        num_runs=1,
        max_labels=21,
        min_labels=5,
        percent_threshold=100
    )

**Subject Sessions Mode:**

Suitable for analyzing the stability of networks within a single subject across multiple sessions.

.. code-block:: python

    pce.applications.icsc_sub_application(
        dataset='subject_sessions',
        data_directory='./data/subject_sessions',
        save_dir='./outputs/subject_level_results',
        num_nodes=264,
        num_threads=1,
        max_labels=21,
        min_labels=5,
        percent_threshold=100
    )


Deep Consensus Clustering (DCC)
-------------------------------

The DCC application implements a deep consensus clustering framework designed to uncover fine-grained subphenotypes from high-dimensional temporal clinical data.

.. code-block:: python

    import pce

    pce.applications.dcc_application(
        input_path='./dataset',
        output_path='./outputs',
        input_dim=26,
        hidden_dims=range(3, 13),
        epoch=5,
        seed=2026
    )
