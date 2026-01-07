=======
PCE API
=======

This reference provides a detailed technical guide to the modules, classes, and functions implemented in PCE, covering the complete workflow from cluster generation to ensemble consensus. It serves as a comprehensive manual for accessing the toolkit’s core utilities, evaluation metrics, and automated experimentation pipelines.

.. currentmodule:: pce

IO
==

This module provides a robust data interface for interacting with MATLAB .mat files and persists experiment results in CSV, Excel, or MAT formats with automatic compatibility for MATLAB v7.3.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   io.load_mat_X_Y
   io.load_mat_BPs_Y
   io.load_rda_X_Y
   io.save_base_mat
   io.save_results_csv
   io.save_results_xlsx
   io.save_results_mat

Generators
==========

The generation module offers diverse strategies to construct base clustering pools from raw feature matrices by controlling random initialization, iterative policies, and subspace perturbations.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   generators.litekmeans
   generators.cdkmeans
   generators.rskmeans
   generators.rpkmeans
   generators.bagkmeans
   generators.hetero_clustering
   generators.sc3_generator
   generators.spectral

Consensus
=========

As the core engine of the toolkit, this module implements a wide range of consensus algorithms (2003–2025) to derive final consensus partitions from base clustering matrices.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    consensus.cspa
    consensus.mcla
    consensus.hgpa
    consensus.ptaal
    consensus.ptacl
    consensus.ptasl
    consensus.ptgp
    consensus.lwea
    consensus.lwgp
    consensus.drec
    consensus.usenc
    consensus.celta
    consensus.ecpcs_hc
    consensus.ecpcs_mc
    consensus.spce
    consensus.trce
    consensus.cdkm
    consensus.mdecbg
    consensus.mdechc
    consensus.mdecsc
    consensus.eccms
    consensus.gtlec
    consensus.kcc_uc
    consensus.kcc_uh
    consensus.ceam
    consensus.space
    consensus.cdec
    consensus.sc3
    consensus.icsc
    consensus.dcc

Metrics
=======

This module provides a comprehensive suite of 14 clustering validation metrics (e.g. ACC, NMI, ARI) for both single-run verification and batch statistical analysis.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    metrics.evaluation_single
    metrics.evaluation_batch

Analysis
========

The analysis module features paper-quality visualization tools, including t-SNE/PCA dimensionality reduction, co-association heatmaps, and parameter sensitivity analysis.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    analysis.plot_2d_scatter
    analysis.plot_coassociation_heatmap
    analysis.plot_metric_line
    analysis.plot_parameter_sensitivity

Pipelines
=========

This module provides high-level automated batch interfaces that streamline the entire "generation-consensus-evaluation" workflow for large-scale comparative experiments.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    pipelines.consensus_batch

Grid
====

Designed for research optimization, this module automates hyperparameter searching across Cartesian product spaces with intelligent pruning and detailed experiment logging.

.. autosummary::
    :toctree: generated/
    :template: class.rst
    :nosignatures:

    grid.GridSearcher

Utils
=====

A utility module for development and debugging, supporting function parameter signature inspection and differentiation between fixed and searchable hyperparameters.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    utils.show_function_params

Applications
============

This module encapsulates end-to-end, domain-specific clustering solutions, integrating established preprocessing and analysis logic for scenarios like single-cell RNA-seq and network science.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    applications.sc3_application
    applications.icsc_mul_application
    applications.icsc_sub_application
    applications.dcc_application
    applications.dcc_cluster_transfer
    applications.dcc_relative_risk
    applications.dcc_comorbidity_bubble
    applications.dcc_KDIGO_circlize
    applications.dcc_survival_analysis
    applications.dcc_KDIGO_dynamic
