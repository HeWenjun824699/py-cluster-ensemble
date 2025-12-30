import time
import os
from typing import Optional, Union, List
import numpy as np
import pandas as pd

from .methods.SC3 import SC3
from .methods.SC3.analysis import organise_de_genes, organise_marker_genes, organise_outliers
from .methods.SC3.export import sc3_export_results
from .methods.SC3.plot import plot_consensus, plot_silhouette, plot_expression, plot_de_genes, plot_markers

def sc3(
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
    SC3 (Single-Cell Consensus Clustering) Wrapper.

    Strict Python port of the SC3 R package.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix, shape (n_samples, n_features).
        Corresponds to the 'counts' or 'logcounts' slot in R.
    Y : np.ndarray, optional
        True labels. Not used by SC3 algorithm (strictly unsupervised).
        Retained in signature only for compatibility with framework interfaces.
    nClusters : int, optional
        Target number of clusters (k).
        If None, estimated automatically using Tracy-Widom theory (sc3_estimate_k).
        Note: Ground truth Y is NEVER used to infer k in this strictly unsupervised implementation.       
    gene_names : list or np.ndarray, optional
        List of gene symbols. If None, generated as 'Gene_0', 'Gene_1'...
    cell_names : list or np.ndarray, optional
        List of cell IDs. If None, generated as 'Cell_0', 'Cell_1'...
    output_directory : str, optional
        If provided, results (DE genes, Markers, Outliers) will be exported to Excel in this directory.   
    gene_filter : bool, default=True
        Whether to perform gene filtering based on dropout percentage.
    pct_dropout_min : int, default=10
        Minimum dropout percentage for gene filtering.
    pct_dropout_max : int, default=90
        Maximum dropout percentage for gene filtering.
    d_region_min : float, default=0.04
    d_region_max : float, default=0.07
    svm_max : int, default=5000
    svm_num_cells : int, optional
    biology : bool, default=False
        Whether to calculate biological features.
    kmeans_nstart : int, default=1000
    kmeans_iter_max : int, default=1e9
    n_cores : int, optional
    seed : int, default=2026

    Returns
    -------
    tuple
        - labels: Predicted labels (np.ndarray).
        - biology: Dictionary containing biological features (dict).
        - time_cost: Execution time (float).
    """

    # Run SC3
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

        # If output_directory is set, force biology calculation to have something to export
        run_biology = biology or (output_directory is not None)

        # nClusters=None will trigger internal estimate_k()
        labels, biology_res = model.run(
            n_clusters=nClusters,
            biology=run_biology,
            kmeans_nstart=kmeans_nstart,
            kmeans_iter_max=kmeans_iter_max
        )

        # Post-processing: Analysis and Export
        if output_directory is not None:
            # Handle Gene Names (Filtering)
            if gene_names is None:
                original_gene_names = np.array([f"Gene_{i}" for i in range(X.shape[1])])
            else:
                original_gene_names = np.array(gene_names)
            
            # The model uses filtered data for plotting, so we need filtered names for plots
            # But for export, we want FULL list.
            if model.gene_mask is not None:
                filtered_gene_names = original_gene_names[model.gene_mask]
            else:
                filtered_gene_names = original_gene_names

            # Handle Cell Names
            if cell_names is None:
                cell_names = np.array([f"Cell_{i}" for i in range(X.shape[0])])
            else:
                cell_names = np.array(cell_names)

            # Determine K
            k = len(np.unique(labels))

            cells_df = pd.DataFrame(index=cell_names)
            genes_df = pd.DataFrame({'feature_symbol': original_gene_names})

            # Add Clusters to Cells DF
            cells_df[f"sc3_{k}_clusters"] = labels + 1 # R uses 1-based indexing

            if biology_res:
                # Organise results (Full Dimensions)
                
                # DE Genes
                de_df = organise_de_genes(biology_res, original_gene_names, k, model.gene_mask)
                if de_df is not None:
                    genes_df = pd.merge(genes_df, de_df, on='feature_symbol', how='left')

                # Marker Genes
                mark_df = organise_marker_genes(biology_res, original_gene_names, k, model.gene_mask)
                if mark_df is not None:
                    genes_df = pd.merge(genes_df, mark_df, on='feature_symbol', how='left')

                # Outliers
                outl_df = organise_outliers(biology_res, cell_names, k)
                if outl_df is not None:
                    # Merge by index
                    cells_df = cells_df.merge(outl_df, left_index=True, right_index=True, how='left')

                # Export Excel
                sc3_export_results(cells_df, genes_df, output_directory)

            # Export Plots (Use Filtered Data)
            try:
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                # 1. Consensus Matrix
                plot_consensus(
                    model.consensus_matrix,
                    labels=labels,
                    file_path=os.path.join(output_directory, "png", "consensus_matrix.png")
                )

                # 2. Silhouette
                plot_silhouette(
                    model.consensus_matrix,
                    labels=labels,
                    file_path=os.path.join(output_directory, "png", "silhouette.png")
                )

                # 3. Gene Expression (Filtered data)
                plot_expression(
                    model.data,
                    labels=labels,
                    file_path=os.path.join(output_directory, "png", "expression.png")
                )

                # 4. DE Genes
                # For plotting, we pass the raw p-values dictionary (which is aligned to FILTERED data)
                if 'de' in biology_res:
                    plot_de_genes(
                        model.data,
                        labels=labels,
                        de_genes_dict=biology_res['de'],
                        file_path=os.path.join(output_directory, "png", "de_genes.png")
                    )

                # 5. Marker Genes
                # For plotting, we pass the raw marker dict (aligned to FILTERED data)
                if 'marker' in biology_res:
                    plot_markers(
                        model.data,
                        labels=labels,
                        marker_res=biology_res['marker'],
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
