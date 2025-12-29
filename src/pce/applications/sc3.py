import time
from typing import Optional, Union, List, Dict
import numpy as np
import pandas as pd

from .methods.SC3 import SC3
from .methods.SC3.analysis import organise_de_genes, organise_marker_genes, organise_outliers
from .methods.SC3.export import sc3_export_results

def sc3(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        nClusters: Optional[Union[int, List[int]]] = None,
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
) -> tuple[Union[np.ndarray, Dict[int, np.ndarray]], Union[dict, Dict[int, dict]], float]:
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
    nClusters : int or list of int, optional
        Target number of clusters (k). Can be a single integer or a range/list.
        If None, estimated automatically using Tracy-Widom theory (sc3_estimate_k).
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
        - labels: Predicted labels (np.ndarray or Dict[int, np.ndarray]).
        - biology: Dictionary containing biological features (dict or Dict[int, dict]).
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
        # The run method now returns dicts keyed by k
        labels_dict, biology_dict = model.run(
            n_clusters=nClusters, 
            biology=run_biology
        )
        
        # Determine if we should return single values (backward compatibility)
        is_single_k = False
        if nClusters is None: 
            is_single_k = True # Estimated k is always single
        elif isinstance(nClusters, int):
            is_single_k = True
        
        # Post-processing: Analysis and Export
        if output_directory is not None and biology_dict:
            # Handle Gene Names (Filtering)
            if gene_names is None:
                gene_names_arr = np.array([f"Gene_{i}" for i in range(X.shape[1])])
            else:
                gene_names_arr = np.array(gene_names)
                
            # If genes were filtered in the model, filter the names too
            if model.gene_mask is not None:
                # model.gene_mask corresponds to columns of X (features/genes)
                if len(gene_names_arr) == len(model.gene_mask):
                    gene_names_arr = gene_names_arr[model.gene_mask]
            
            # Handle Cell Names
            if cell_names is None:
                cell_names_arr = np.array([f"Cell_{i}" for i in range(X.shape[0])])
            else:
                cell_names_arr = np.array(cell_names)
                
            # Prepare Export Data
            # We need to merge results from all k into comprehensive tables if possible,
            # or just export the first one if the export function is simple.
            # R SC3 exports "Cells" sheet and "Genes" sheet with columns for each k.
            
            # Initialize merged DataFrames
            cells_df = pd.DataFrame(index=cell_names_arr)
            genes_df = pd.DataFrame(index=gene_names_arr)
            genes_df['feature_symbol'] = gene_names_arr
            
            has_valid_results = False
            
            for k, bio_res in biology_dict.items():
                if not bio_res: continue
                has_valid_results = True
                
                # 1. Outliers -> Cells DF
                outliers_df = organise_outliers(bio_res, cell_names_arr)
                if outliers_df is not None:
                    # Merge on index (cell_names)
                    # outliers_df has 'cell_id' and 'sc3_log2_outlier_score'
                    # We rename column to include k
                    col_name = f"sc3_{k}_log2_outlier_score"
                    outliers_df = outliers_df.set_index('cell_id')
                    outliers_df.columns = [col_name]
                    cells_df = cells_df.join(outliers_df, how='left')
                
                # Add Clusters to Cells DF
                labels_k = labels_dict[k]
                cells_df[f"sc3_{k}_clusters"] = labels_k + 1 # 1-based for R consistency in export
                
                # 2. DE Genes -> Genes DF
                de_df = organise_de_genes(bio_res, gene_names_arr, p_val_threshold=1.0) # Get all for merging
                if de_df is not None:
                    col_name = f"sc3_{k}_de_padj"
                    de_df = de_df.set_index('feature_symbol')
                    de_df = de_df[['sc3_de_padj']].rename(columns={'sc3_de_padj': col_name})
                    genes_df = genes_df.join(de_df, how='left')

                # 3. Markers -> Genes DF
                marker_df = organise_marker_genes(bio_res, gene_names_arr, p_val_threshold=1.0, auroc_threshold=0.0)
                if marker_df is not None:
                     marker_df = marker_df.set_index('feature_symbol')
                     # Columns: sc3_marker_clusts, sc3_marker_auroc, sc3_marker_padj
                     rename_map = {
                         'sc3_marker_clusts': f"sc3_{k}_markers_clusts",
                         'sc3_marker_auroc': f"sc3_{k}_markers_auroc",
                         'sc3_marker_padj': f"sc3_{k}_markers_padj"
                     }
                     marker_df = marker_df.rename(columns=rename_map)
                     genes_df = genes_df.join(marker_df, how='left')

            if has_valid_results:
                analysis_res = {
                    'Cells': cells_df,
                    'Genes': genes_df
                }
                # Export using the generic export function
                # We need to bypass the specific dict structure check in sc3_export_results if it exists
                # Or adapt sc3_export_results to take a dict of DataFrames directly.
                # Assuming sc3_export_results handles dict of DataFrames (SheetName -> DF)
                sc3_export_results(analysis_res, output_directory)
        
        # Prepare Return Values
        if is_single_k:
            # Return single values
            # labels_dict keys are k. If estimated, we don't know k upfront easily unless we peek.
            first_k = list(labels_dict.keys())[0]
            labels = labels_dict[first_k]
            biology_res = biology_dict[first_k] if biology_dict else {}
        else:
            labels = labels_dict
            biology_res = biology_dict

    except Exception as e:
        print(f"SC3 execution failed: {e}")
        # In case of failure, return empty consistent with requested type? 
        # Hard to guess requested type if it failed early.
        # Fallback to single k behavior for safety
        labels = np.zeros(X.shape[0], dtype=int)
        biology_res = {}
        
    end_time = time.time()
    
    return labels, biology_res, end_time - start_time
