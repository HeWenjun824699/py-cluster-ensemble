import numpy as np
import pandas as pd

def organise_de_genes(biology_res, gene_names, p_val_threshold=0.05):
    """
    Organise DE genes results.
    R equivalent: organise_de_genes
    
    Parameters
    ----------
    biology_res : dict
        Raw biology results from SC3.run()
    gene_names : list or np.ndarray
        Names of genes corresponding to columns of X.
    p_val_threshold : float
        Threshold for adjusted p-value.
        
    Returns
    -------
    pd.DataFrame
        Filtered and sorted DE genes.
    """
    p_values = biology_res.get('de')
    if p_values is None or len(p_values) == 0:
        return None
    
    # Ensure gene_names length matches p_values
    if len(gene_names) != len(p_values):
        # Fallback if filtered data was used but names correspond to original?
        # In SC3 class gene_filter might drop genes.
        # The gene_names passed to sc3() are likely original.
        # We need to handle this. For now assume matching length or truncate.
        # Ideally sc3 wrapper should handle filtering of names too.
        # But SC3 class handles filtering internally.
        pass
        
    df = pd.DataFrame({
        'feature_symbol': gene_names,
        'sc3_de_padj': p_values
    })
    
    # Filter and sort
    df = df[df['sc3_de_padj'] < p_val_threshold]
    df = df.sort_values('sc3_de_padj')
    
    return df

def organise_marker_genes(biology_res, gene_names, p_val_threshold=0.05, auroc_threshold=0.85):
    """
    Organise Marker genes results.
    R equivalent: organise_marker_genes
    
    Parameters
    ----------
    biology_res : dict
    gene_names : list
    p_val_threshold : float
    auroc_threshold : float
    
    Returns
    -------
    pd.DataFrame
    """
    marker_res = biology_res.get('marker')
    if marker_res is None or len(marker_res.get('pvalue', [])) == 0:
        return None
        
    df = pd.DataFrame({
        'feature_symbol': gene_names,
        'sc3_marker_clusts': marker_res['clusts'],
        'sc3_marker_auroc': marker_res['auroc'],
        'sc3_marker_padj': marker_res['pvalue']
    })
    
    # Filter
    mask = (df['sc3_marker_padj'] < p_val_threshold) & \
           (df['sc3_marker_auroc'] > auroc_threshold) & \
           (pd.notna(df['sc3_marker_clusts']))
           
    df = df[mask]
    
    # Sort by cluster then by AUROC (descending)
    df = df.sort_values(by=['sc3_marker_clusts', 'sc3_marker_auroc'], ascending=[True, False])
    
    return df

def organise_outliers(biology_res, cell_names=None):
    """
    Organise outlier cells results.
    """
    outl = biology_res.get('outl')
    if outl is None or len(outl) == 0:
        return None
    
    n_cells = len(outl)
    if cell_names is None:
        cell_names = [f"Cell_{i}" for i in range(n_cells)]
        
    df = pd.DataFrame({
        'cell_id': cell_names,
        'sc3_log2_outlier_score': np.log2(outl + 1)
    })
    
    return df
