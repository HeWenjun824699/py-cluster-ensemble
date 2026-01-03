==================
Evaluation Metrics
==================

[cite_start]PCE provides a suite of 14 clustering validation metrics to quantify the performance of ensemble results[cite: 8].

Key Metrics
-----------

The toolkit supports both external validation (requiring ground truth) and statistical consistency measures:

* **Accuracy (ACC)**: Measures the maximum matching between predicted and true labels.
* **Normalized Mutual Information (NMI)**: A standard information-theoretic measure.
* **Adjusted Rand Index (ARI)**: Corrects the Rand Index for chance.
* [cite_start]**Other Metrics**: Includes Purity, F-score, Precision, Recall, Entropy, and specialized metrics like SDCS and RME[cite: 8].

Batch Evaluation
----------------

In research, results from a single run are often insufficient. [cite_start]PCE's ``evaluation_batch`` is designed for statistical significance[cite: 8].

* **Input**: A list of partition labels from multiple experiment runs.
* [cite_start]**Output**: A list of dictionaries containing metrics for each run, which can be easily summarized into mean and standard deviation for paper tables[cite: 8].

Integration with Pipelines
--------------------------

[cite_start]These metrics are fully integrated into the ``pce.pipelines.consensus_batch`` and ``pce.grid.GridSearcher`` modules, enabling fully automated performance reporting across entire datasets[cite: 10, 11].