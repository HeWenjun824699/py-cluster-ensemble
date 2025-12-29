import time
from typing import Optional, Union, List
import numpy as np

from .methods.SC3 import SC3
from .methods.SC3.analysis import organise_de_genes, organise_marker_genes, organise_outliers
from .methods.SC3.export import sc3_export_results

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
        if output_directory is not None and biology_res:
            # Handle Gene Names (Filtering)
            if gene_names is None:
                gene_names = np.array([f"Gene_{i}" for i in range(X.shape[1])])
            else:
                gene_names = np.array(gene_names)
                
            # If genes were filtered in the model, filter the names too
            if model.gene_mask is not None:
                # model.gene_mask corresponds to columns of X (features/genes)
                if len(gene_names) == len(model.gene_mask):
                    gene_names = gene_names[model.gene_mask]
            
            # Handle Cell Names
            if cell_names is None:
                cell_names = np.array([f"Cell_{i}" for i in range(X.shape[0])])
            else:
                cell_names = np.array(cell_names)
                
            # Organise results
            analysis_res = {}
            analysis_res['de_genes'] = organise_de_genes(biology_res, gene_names)
            analysis_res['marker_genes'] = organise_marker_genes(biology_res, gene_names)
            analysis_res['outliers'] = organise_outliers(biology_res, cell_names)
            
            # Export
            sc3_export_results(analysis_res, output_directory)
        
    except Exception as e:
        print(f"SC3 execution failed: {e}")
        labels = np.zeros(X.shape[0], dtype=int)
        biology_res = {}
        
    end_time = time.time()
    
    return labels, biology_res, end_time - start_time