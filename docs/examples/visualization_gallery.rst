=====================
Visualization Gallery
=====================

PCE provides paper-quality visualization tools to help researchers interpret clustering results and analyze ensemble structures. These tools are designed for academic publication, supporting high-resolution bitmap (PNG) and vector graphics (PDF) outputs.

Dimensionality Reduction
------------------------

Visualize high-dimensional data distributions using t-SNE or PCA. This is essential for showcasing raw data structures or verifying the clustering effect in a reduced 2D space.

.. code-block:: python

   import pce.io as io
   import pce.analysis as ana

   # 1. Load raw data
   X, Y = io.load_mat_X_Y('./data/sample.mat')

   # 2. Generate t-SNE scatter plot
   ana.plot_2d_scatter(
       X, Y,
       xlabel='Dimension 1',
       ylabel='Dimension 2',
       method='tsne',
       title='Ground Truth Visualization (t-SNE)',
       save_path='./results/png/X_Y_tsne.png'
   )

Co-association Heatmap
----------------------

Analyze the consistency of the ensemble by plotting a co-association heatmap. By sorting samples based on labels, the heatmap reveals the underlying block structure discovered by the base partitions.

.. code-block:: python

   import pce.io as io
   import pce.analysis as ana

   # 1. Load precomputed base partitions
   BPs, Y = io.load_mat_BPs_Y('./data/CDKM200/sample_CDKM200.mat')

   # 2. Plot co-association matrix heatmap
   ana.plot_coassociation_heatmap(
       BPs, Y,
       xlabel='Sample Index (Sorted by Ground Truth)',
       ylabel='Sample Index (Sorted by Ground Truth)',
       title='Ensemble Consensus Matrix Heatmap',
       save_path='./results/png/BPs_Y_heatmap.png'
   )

Metric Stability (Trace Plot)
-----------------------------

Visualize the stability of clustering performance across multiple experiment runs. This helps researchers assess the reliability and variance of various metrics.

.. code-block:: python

   import json
   import pce.analysis as ana

   # 1. Load batch evaluation results (typically from metrics.json)
   with open('./results/grid/Exp_001_LWEA/metrics.json') as f:
       results_list = json.loads(f.read())

   # 2. Define metrics to be displayed
   metrics = ["ACC", "NMI", "Purity", "AR", "RI", "F-Score", "Precision", "Recall", "RME", "Bal"]

   # 3. Plot metric line chart over multiple runs
   ana.plot_metric_line(
       results_list=results_list,
       metrics=metrics,
       xlabel='Experiment Run ID',
       ylabel='Score',
       title='Clustering Performance Stability',
       save_path='./results/png/line_plot.png'
   )

Sensitivity Analysis
--------------------

After running a Grid Search, you can visualize how a specific hyperparameter (e.g., ``theta`` or ``lamb``) affects the performance of the algorithm using the "control variables" method.

.. code-block:: python

   import pce.analysis as ana

   # Path to the summary CSV from Grid Search
   csv_file = './results/grid/grid_search_summary.csv'

   # Plot sensitivity of parameter 'theta' on Accuracy
   ana.plot_parameter_sensitivity(
       csv_file,
       target_param='theta',
       metric='ACC',
       fixed_params={},    # Optionally fix other parameters
       method_name='lwea',
       save_path='./results/png/parameter_sensitivity.png',
       show=True,
       show_values=True
   )
