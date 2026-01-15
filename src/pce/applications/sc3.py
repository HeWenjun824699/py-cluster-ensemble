import time
import os
from typing import Optional, Union, List
import numpy as np
import pandas as pd

from .methods.SC3 import SC3
from .methods.SC3.analysis import organise_de_genes, organise_marker_genes, organise_outliers
from .methods.SC3.export import sc3_export_results
from .methods.SC3.plot import plot_consensus, plot_silhouette, plot_expression, plot_de_genes, plot_markers

def sc3_application(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[int] = None,
        gene_names: Optional[Union[List[str], np.ndarray]] = None,
        cell_names: Optional[Union[List[str], np.ndarray]] = None,
        output_directory: Optional[str] = None,
        gene_filter: bool = True,
        pct_dropout_min: int = 10,
        pct_dropout_max: int = 90,
        d_region_min: float = 0.04,
        d_region_max: float = 0.07,
        svm_max: int = 5000,
        svm_num_cells: Optional[int] = None,
        biology: bool = False,
        kmeans_nstart: int = 1000,
        kmeans_iter_max: int = 1000000000,
        n_cores: Optional[int] = None,
        seed: int = 2026
) -> tuple[np.ndarray, dict, float]:
    """
    Execute the SC3 (Single-Cell Consensus Clustering) pipeline on single-cell RNA-seq data.

    This function serves as a Python wrapper for the SC3 algorithm originally published 
    in Nature Methods (2017). It implements a consensus clustering framework tailored 
    for single-cell expression data, including gene filtering, dimensionality reduction, 
    consensus aggregation, and optional downstream biological analysis.

    Parameters
    ----------
    X : np.ndarray
        Input gene expression matrix of shape (n_samples, n_features), where `n_samples` 
        represents cells and `n_features` represents genes. This corresponds to the 
        'counts' or 'logcounts' matrix in the original R package.
    Y : np.ndarray, optional
        True class labels for the samples. This parameter is not utilized by the 
        unsupervised SC3 algorithm and is included primarily for interface consistency. 
        Default is None.
    nClusters : int, optional
        The target number of clusters (k). If None, the optimal number of clusters 
        is automatically estimated using the Tracy-Widom theory. Default is None.
    gene_names : list of str or np.ndarray, optional
        A list or array of gene symbols corresponding to the columns of X. If None, 
        names are generated as 'Gene_0', 'Gene_1', etc. Required for meaningful 
        biological analysis reports (DE/Markers). Default is None.
    cell_names : list of str or np.ndarray, optional
        A list or array of cell identifiers corresponding to the rows of X. If None, 
        names are generated as 'Cell_0', 'Cell_1', etc. Default is None.
    output_directory : str, optional
        Filesystem path to save analysis results. If provided, the function exports 
        an Excel report (containing DE genes, marker genes, and outlier scores) and 
        generates visualization plots (PNG format). Default is None.
    gene_filter : bool, default=True
        If True, performs gene filtering based on dropout percentages to remove 
        rare or ubiquitous genes.
    pct_dropout_min : int, default=10
        Minimum dropout percentage. Genes expressed in fewer than this percentage 
        of cells are filtered out.
    pct_dropout_max : int, default=90
        Maximum dropout percentage. Genes expressed in more than this percentage 
        of cells (ubiquitous genes) are filtered out.
    d_region_min : float, default=0.04
        The lower bound of the interval for the number of eigenvectors (expressed 
        as a fraction of total cells) used in spectral clustering.
    d_region_max : float, default=0.07
        The upper bound of the interval for the number of eigenvectors used in 
        spectral clustering.
    svm_max : int, default=5000
        The maximum number of cells allowed before switching to SVM-based prediction. 
        If the number of cells exceeds this threshold, a subset is used for clustering, 
        and the rest are predicted using a Support Vector Machine.
    svm_num_cells : int, optional
        The specific number of cells to use for the training set when SVM prediction 
        mode is triggered (i.e., when n_samples > svm_max). If None, a default 
        heuristic is used. Default is None.
    biology : bool, default=False
        If True, computes biological features including differential expression (DE) 
        genes, marker genes, and cell outlier scores.
    kmeans_nstart : int, default=1000
        The number of random initializations (restarts) for the K-means algorithm 
        to maximize the probability of finding the global optimum.
    kmeans_iter_max : int, default=1000000000
        The maximum number of iterations allowed for a single run of the K-means 
        algorithm.
    n_cores : int, optional
        The number of CPU cores to utilize for parallel computation, particularly 
        for distance matrix calculations. If None, sequential processing is used. 
        Default is None.
    seed : int, default=2026
        The random seed used for initializing K-means and random sampling, ensuring 
        reproducibility of results.

    Returns
    -------
    labels : np.ndarray
        An array of shape (n_samples,) containing the predicted cluster labels 
        for each cell.
    biology_res : dict
        A dictionary containing calculated biological statistics, including:
        - 'de': Differential expression results.
        - 'marker': Marker gene results.
        - 'outlier': Outlier scores.
        Returns an empty dictionary if `biology` is False and `output_directory` is None.
    time_cost : float
        The total wall-clock time consumed by the execution in seconds.

    Notes
    -----
    The biological analysis (DE, markers, outliers) is computationally intensive. 
    It is automatically enabled if `output_directory` is specified, regardless of 
    the `biology` parameter value, to ensure there is data to export.
    """

    # Initialize SC3 model (Nature Methods, 2017)
    start_time = time.time()
    try:
        model = SC3(
            data=X,
            gene_filter=gene_filter,
            pct_dropout_min=pct_dropout_min,
            pct_dropout_max=pct_dropout_max,
            d_region_min=d_region_min,
            d_region_max=d_region_max,
            svm_max=svm_max,
            svm_num_cells=svm_num_cells,
            n_cores=n_cores,
            seed=seed
        )

        # Force biology computation if output directory is specified to ensure exportable results
        run_biology = biology or (output_directory is not None)

        # Run the model (nClusters=None triggers automatic k estimation via Tracy-Widom theory)
        labels, biology_res = model.run(
            n_clusters=nClusters,
            biology=run_biology,
            kmeans_nstart=kmeans_nstart,
            kmeans_iter_max=kmeans_iter_max
        )

        # Post-processing: Analysis and Export
        if output_directory is not None:
            # Handle Gene Names (for export vs plotting)
            if gene_names is None:
                original_gene_names = np.array([f"Gene_{i}" for i in range(X.shape[1])])
            else:
                original_gene_names = np.array(gene_names)
            
            # The model uses filtered data for plotting, so we need names aligned to the filtered matrix.
            # However, for the exported Excel report, we want the full list of genes.
            if model.gene_mask is not None:
                filtered_gene_names = original_gene_names[model.gene_mask]
            else:
                filtered_gene_names = original_gene_names

            # Handle Cell Names
            if cell_names is None:
                cell_names = np.array([f"Cell_{i}" for i in range(X.shape[0])])
            else:
                cell_names = np.array(cell_names)

            # Determine K (number of clusters found)
            k = len(np.unique(labels))

            cells_df = pd.DataFrame(index=cell_names)
            genes_df = pd.DataFrame({'feature_symbol': original_gene_names})

            # Add gene filter to Genes DataFrame
            if model.gene_mask is not None:
                genes_df['sc3_gene_filter'] = model.gene_mask
            else:
                genes_df['sc3_gene_filter'] = True

            # Add Cluster assignments to Cells DataFrame (convert to 1-based indexing for R consistency)
            cells_df[f"sc3_{k}_clusters"] = labels + 1

            # Biological Results
            mark_df = None
            de_df = None
            outl_df = None

            if biology_res:
                # Organise biological results (mapped to full dimensions)

                # Marker Genes
                mark_df = organise_marker_genes(biology_res, original_gene_names, k, model.gene_mask)
                if mark_df is not None:
                    genes_df = pd.merge(genes_df, mark_df, on='feature_symbol', how='left')

                # Differential Expression (DE) Genes
                de_df = organise_de_genes(biology_res, original_gene_names, k, model.gene_mask)
                if de_df is not None:
                    genes_df = pd.merge(genes_df, de_df, on='feature_symbol', how='left')

                # Outlier Scores
                outl_df = organise_outliers(biology_res, cell_names, k)
                if outl_df is not None:
                    # Merge by index (cell names)
                    cells_df = cells_df.merge(outl_df, left_index=True, right_index=True, how='left')

                # Export results to Excel
                sc3_export_results(cells_df, genes_df, output_directory)

            # Generate and Export Visualization Plots (using Filtered Data)
            try:
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                # 1. Consensus Matrix Plot
                plot_consensus(
                    model.consensus_matrix,
                    labels=labels,
                    file_path=os.path.join(output_directory, "png", "consensus_matrix.png")
                )

                # 2. Silhouette Plot
                plot_silhouette(
                    model.consensus_matrix,
                    labels=labels,
                    file_path=os.path.join(output_directory, "png", "silhouette.png")
                )

                # 3. Gene Expression Plot (on filtered data)
                plot_expression(
                    model.data,
                    labels=labels,
                    file_path=os.path.join(output_directory, "png", "expression_py.png")
                )

                # 4. DE Genes Heatmap
                # Pass the raw DE p-values dictionary (aligned to filtered data)
                if 'de' in biology_res:
                    plot_de_genes(
                        data=X,
                        labels=labels,
                        de_results_df=de_df,
                        consensus_matrix=model.consensus_matrix,
                        file_path=os.path.join(output_directory, "png", "de_genes.png")
                    )

                # 5. Marker Genes Heatmap
                # Pass the raw marker dictionary (aligned to filtered data)
                if 'marker' in biology_res:
                    plot_markers(
                        data=X,
                        labels=labels,
                        marker_res=mark_df,
                        consensus_matrix=model.consensus_matrix,
                        file_path=os.path.join(output_directory, "png", "marker_genes.png")
                    )

            except Exception as plot_e:
                print(f"Plot generation failed: {plot_e}")

    except Exception as e:
        print(f"SC3 execution failed: {e}")
        import traceback
        traceback.print_exc()
        labels = np.zeros(X.shape[0], dtype=int)
        biology_res = {}

    end_time = time.time()

    return labels, biology_res, end_time - start_time
