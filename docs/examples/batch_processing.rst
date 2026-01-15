=============================
Automated Batch Processing
=============================

This tutorial covers how to handle multiple datasets efficiently using PCE's high-level pipelines. Depending on the level of control required for your research, you can choose between a fully automated pipeline or a modular step-by-step workflow.

Automated Pipeline (Recommended)
--------------------------------

The ``consensus_batch`` pipeline is designed for large-scale experiments. It automatically scans a directory for ``.mat`` files, generates base partitions if they are missing, runs the ensemble algorithm, and exports results to CSV, Excel, or MAT formats.

.. code-block:: python

    import pce

    # Set your data and output directories
    data_dir = "./data/CDKM200"
    output_dir = "./results/workflow/batch"

    # Execute the automated pipeline for multiple datasets with a single algorithm
    pce.pipelines.consensus_batch(
       input_dir=data_dir,
       output_dir=output_dir,
       save_format="csv",
       consensus_method="cspa",
       seed=2026,
       nRepeat=10,
       overwrite=True
    )

    # # Advanced Usage: Iterate through multiple datasets and multiple algorithms
    # consensus_methods = [
    #     "cspa", "mcla", "hgpa", "ptaal", "ptacl", "ptasl", "ptgp",
    #     "lwea", "lwgp", "drec", "usenc", "celta", "ecpcs_hc",
    #     "ecpcs_mc", "spce", "trce", "cdkm", "mdecbg", "mdechc",
    #     "mdecsc", "eccms", "gtlec", "kcc_uc", "kcc_uh", "ceam",
    #     "space", "cdec", "sc3", "icsc", "dcc"
    # ]
    #
    # for method in consensus_methods:
    #     pce.pipelines.consensus_batch(
    #         input_dir=data_dir,
    #         output_dir=output_dir,
    #         save_format="csv",
    #         consensus_method=method,
    #         seed=2026,
    #         nRepeat=10,
    #         overwrite=True
    #     )

Modular Step-by-Step Workflow
-----------------------------

For researchers who need to intervene at specific stages (e.g., custom data preprocessing or manual result inspection), PCE allows manual modular calls.

1. **Loading Data**: PCE handles MATLAB v7.3 compatibility and 1-based index correction automatically.
2. **Generating Pools**: Create a diverse pool of clusterings using strategies like Coordinate Descent K-Means.
3. **Running Consensus**: Derive final labels using standard algorithms.
4. **Evaluating**: Calculate 14 different metrics including ACC and NMI.

.. code-block:: python

    import numpy as np
    import pce

    # 1. Load data
    data_path = "./data/isolet_1560n_617d_2c.mat"
    X, Y = pce.io.load_mat_X_Y(data_path)

    # 2. Generate Base Partitions (e.g., using CD-Means)
    BPs = pce.generators.cdkmeans(X, Y)

    # Optional: Save generated bases
    pce.io.save_base_mat(BPs, Y, "./results/workflow/single/isolet_CDKM200.mat")

    # 3. Execute Consensus
    # Explicitly calculate the number of target clusters
    nClusters = len(np.unique(Y))
    labels_list, time_list = pce.consensus.cspa(BPs, nClusters=nClusters)

    # 4. Evaluation and Persistence
    res = pce.metrics.evaluation_batch(labels_list, Y, time_list)

    # 5. Save the results to CSV for analysis
    pce.io.save_results_csv(res, "./results/workflow/single/isolet_CDKM200_CSPA.csv")

