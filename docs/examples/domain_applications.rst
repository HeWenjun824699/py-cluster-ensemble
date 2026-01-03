===================
Domain Applications
===================

[cite_start]The ``applications`` module encapsulates end-to-end solutions for specific research fields[cite: 13].

Single-Cell RNA-Seq (SC3)
-------------------------

[cite_start]PCE includes a Python implementation of the SC3 algorithm for single-cell consensus clustering[cite: 13]. It integrates gene filtering and biological analysis.

.. code-block:: python

   from pce.applications import sc3

   # Run SC3 on expression data
   labels, bio_res, time_cost = sc3(
       X,
       gene_names=my_genes,
       biology=True,
       output_directory='./sc3_output'
   )

Network Science (FastEnsemble)
------------------------------

[cite_start]For large-scale graph data, PCE provides the ``fast_ensemble`` interface for scalable community detection[cite: 13].

.. code-block:: python

   from pce.applications import fast_ensemble

   # Execute fast ensemble on an edge-list file
   fast_ensemble(
       input_file='network.txt',
       output_file='communities.csv',
       algorithm='leiden-cpm'
   )