import numpy as np
import pandas as pd


def organise_de_genes(biology_res, gene_names, k, gene_mask=None):
    """
    Organise DE genes results matching R SC3-Nature methods-2017 export format.
    
    Parameters
    ----------
    biology_res : dict
        Raw biology results from SC3-Nature methods-2017.run()
    gene_names : list or np.ndarray
        Original list of gene names (length = n_original_genes).
    k : int
        Number of clusters.
    gene_mask : np.ndarray, optional
        Boolean mask used for gene filtering. 
        If None, assumes no filtering was done.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [feature_symbol, sc3_{k}_de_padj]
        Includes ALL genes. Filtered-out genes have NA.
    """
    p_values = biology_res.get('de')
    if p_values is None:
        return None

    n_genes = len(gene_names)
    full_p_values = np.full(n_genes, np.nan)

    if gene_mask is not None:
        # p_values corresponds to True values in gene_mask
        if np.sum(gene_mask) == len(p_values):
            full_p_values[gene_mask] = p_values
        else:
            # Dimension mismatch safeguard
            pass
    else:
        full_p_values = p_values

    col_name = f"sc3_{k}_de_padj"
    
    df = pd.DataFrame({
        'feature_symbol': gene_names,
        col_name: full_p_values
    })

    return df


def organise_marker_genes(biology_res, gene_names, k, gene_mask=None):
    """
    Organise Marker genes results matching R SC3-Nature methods-2017 export format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 
        [feature_symbol, sc3_{k}_markers_clusts, sc3_{k}_markers_auroc, sc3_{k}_markers_padj]
    """
    marker_res = biology_res.get('marker')
    if marker_res is None:
        return None

    n_genes = len(gene_names)
    
    # Initialize full arrays with NaNs
    full_clusts = np.full(n_genes, np.nan)
    full_auroc = np.full(n_genes, np.nan)
    full_pvalue = np.full(n_genes, np.nan)

    # Extract calculated values
    # In Python code, get_marker_genes returns dict/df aligned with filtered data
    # Convert to 1-based indexing for R consistency
    calc_clusts = marker_res['clusts'] + 1
    calc_auroc = marker_res['auroc']
    calc_pvalue = marker_res['pvalue']

    if gene_mask is not None:
        if np.sum(gene_mask) == len(calc_clusts):
            full_clusts[gene_mask] = calc_clusts
            full_auroc[gene_mask] = calc_auroc
            full_pvalue[gene_mask] = calc_pvalue
    else:
        full_clusts = calc_clusts
        full_auroc = calc_auroc
        full_pvalue = calc_pvalue

    df = pd.DataFrame({
        'feature_symbol': gene_names,
        f"sc3_{k}_markers_clusts": full_clusts,
        f"sc3_{k}_markers_padj": full_pvalue,
        f"sc3_{k}_markers_auroc": full_auroc,
    })

    return df


def organise_outliers(biology_res, cell_names, k):
    """
    Organise outlier cells results.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [sc3_{k}_log2_outlier_score]
        Indexed by cell_names if merged later, but here just returns the column.
    """
    outl = biology_res.get('outl')
    if outl is None:
        return None
    
    col_name = f"sc3_{k}_log2_outlier_score"
    
    df = pd.DataFrame({
        col_name: outl
    }, index=cell_names)

    return df
