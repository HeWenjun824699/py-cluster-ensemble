=====================
Visualization Gallery
=====================

[cite_start]PCE provides paper-quality visualization tools to help interpret clustering results and ensemble structures[cite: 9].

Dimensionality Reduction
------------------------

[cite_start]Visualize high-dimensional data distributions using t-SNE or PCA[cite: 9].

.. code-block:: python

   import pce.analysis as ana

   # Generate t-SNE scatter plot
   ana.plot_2d_scatter(X, Y, method='tsne', title='Data Distribution', save_path='tsne.pdf')

Co-association Heatmap
----------------------

[cite_start]Analyze the consistency of the ensemble by plotting a co-association heatmap[cite: 9]. This reveals the underlying structure discovered by the base partitions.

.. code-block:: python

   # Plot co-association matrix
   ana.plot_coassociation_heatmap(BPs, Y, save_path='heatmap.png')



Sensitivity Analysis
--------------------

[cite_start]After running a Grid Search, you can visualize how a specific hyperparameter affects performance[cite: 9].

.. code-block:: python

   # Plot sensitivity of parameter 't'
   ana.plot_parameter_sensitivity(
       csv_file='grid_results/summary.csv',
       target_param='t',
       metric='ACC',
       save_path='sensitivity.png'
   )