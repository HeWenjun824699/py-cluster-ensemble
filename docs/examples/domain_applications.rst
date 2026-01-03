===================
Domain Applications
===================

The ``applications`` module encapsulates end-to-end solutions for specific research fields.

Single-Cell RNA-Seq (SC3)
-------------------------

PCE includes a Python implementation of the SC3 algorithm for single-cell consensus clustering. It integrates gene filtering and biological analysis.

.. code-block:: python

    from pce.io.load_rda import load_rda_X_Y
    from pce.applications import sc3

    # 1. Load data
    X, Y, gene_names, cell_names = load_rda_X_Y("./data/yan.rda")
    
    # 2.Run SC3 on expression data
    labels, biology_res, time_cost = sc3(
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

Network Science (FastEnsemble)
------------------------------

For large-scale graph data, PCE provides the ``fast_ensemble`` interface for scalable community detection.

.. code-block:: python

    from pce.applications import fast_ensemble
    
    # Execute fast ensemble on an edge-list file
    time_taken = fast_ensemble(
        input_file="data/sc_1.0_ring_cliques_100_10.tsv",
        output_file="results/results.csv",
        n_partitions=10,
        threshold=0.8,
        resolution=0.01,
        algorithm='leiden-cpm',
        relabel=False,
        delimiter=','
    )
