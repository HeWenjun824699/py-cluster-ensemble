==================
Evaluation Metrics
==================

PCE provides a comprehensive suite of **14 clustering validation metrics** to quantify the performance of ensemble results from multiple dimensions. These metrics allow researchers to evaluate the quality of consensus partitions against ground truth or analyze the statistical properties of the clustering structure.

Key Metrics
-----------

The toolkit supports both external validation (requiring ground truth labels) and information-theoretic measures:

* **Accuracy (ACC)**: Measures the maximum matching between predicted cluster assignments and true labels.
* **Normalized Mutual Information (NMI)**: A standard information-theoretic measure that quantifies the mutual information between two partitions, normalized to a range of [0, 1].
* **Adjusted Rand Index (ARI)**: An improved version of the Rand Index that is corrected for chance, providing a more robust measure for cluster similarity.
* **Other Supported Metrics**: The toolkit also calculates Purity, F-score, Precision, Recall, RI (Rand Index), MI (Mutual Information), HI (Hubert Index), Entropy, and specialized research metrics such as **SDCS**, **RME**, and **Bal** (Balance).

Batch Evaluation for Research
-----------------------------

In academic research, results from a single run are often insufficient due to the stochastic nature of many clustering algorithms. PCE's ``evaluation_batch`` function is specifically designed to provide statistical significance for your experiments.

* **Input**: A list of partition labels obtained from multiple experiment repetitions (e.g., from 10 or 20 runs).
* **Output**: A list of structured dictionaries containing all 14 metrics for each individual run.
* **Statistical Summary**: These results can be easily passed to ``pce.io.save_results_xlsx`` or ``pce.io.save_results_csv``, which automatically calculate the **Mean** and **Standard Deviation** for each metric—ideal for direct insertion into academic paper tables.

Integration with Automated Pipelines
------------------------------------

The evaluation module is a core component of PCE's automation ecosystem:

* **Pipelines**: The ``pce.pipelines.consensus_batch`` interface automatically invokes the evaluation module after each consensus task.
* **Grid Search**: The ``pce.grid.GridSearcher`` uses these metrics to rank different hyperparameter combinations, helping you identify the configuration that yields the highest NMI or ACC.