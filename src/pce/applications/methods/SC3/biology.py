import numpy as np
import warnings
from scipy.stats import kruskal, mannwhitneyu, rankdata, chi2
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet

def p_adjust_holm(p_values):
    """
    Holm-Bonferroni correction.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    # Check for NaNs
    mask = ~np.isnan(p_values)
    if not np.any(mask):
        return p_values
        
    p_valid = p_values[mask]
    n_valid = len(p_valid)
    
    idx = np.argsort(p_valid)
    sorted_p = p_valid[idx]
    
    corrections = np.arange(n_valid, 0, -1)
    adjusted = sorted_p * corrections
    
    for i in range(1, n_valid):
        adjusted[i] = max(adjusted[i], adjusted[i-1])
        
    adjusted = np.minimum(adjusted, 1.0)
    
    result = np.full_like(p_values, np.nan)
    result_valid = np.empty_like(p_valid)
    result_valid[idx] = adjusted
    result[mask] = result_valid
    
    return result

def get_de_genes(data, labels):
    """
    Find differentially expressed genes (Kruskal-Wallis).
    
    Parameters
    ----------
    data : np.ndarray
        (n_cells, n_genes)
    labels : np.ndarray
        (n_cells,)
        
    Returns
    -------
    np.ndarray
        (n_genes,) Adjusted p-values.
    """
    n_genes = data.shape[1]
    unique_labels = np.unique(labels)
    p_values = np.ones(n_genes)
    
    if len(unique_labels) < 2:
        return p_values
        
    # Pre-calculate groups indices to speed up loop
    groups_indices = [np.where(labels == l)[0] for l in unique_labels]
    
    for i in range(n_genes):
        gene_expr = data[:, i]
        groups = [gene_expr[idx] for idx in groups_indices]
        
        # Check if all groups have constant values (or all same constant)
        # kruskal requires variance
        try:
            # Check if all values are identical across all groups
            if np.all(gene_expr == gene_expr[0]):
                p_values[i] = 1.0
                continue
                
            stat, p = kruskal(*groups)
            p_values[i] = p
        except ValueError:
            p_values[i] = 1.0
            
    # Adjust p-values (Holm)
    p_values = p_adjust_holm(p_values)
    
    return p_values

def get_marker_genes(data, labels):
    """
    Find marker genes (AUROC + Wilcoxon).
    
    Returns
    -------
    dict
        {'auroc': array, 'clusts': array, 'pvalue': array}
    """
    n_genes = data.shape[1]
    unique_labels = np.unique(labels)
    
    auroc_arr = np.zeros(n_genes)
    clusts_arr = np.zeros(n_genes) # Stores label of the marker cluster
    p_values = np.ones(n_genes)
    
    for i in range(n_genes):
        gene_expr = data[:, i]
        ranks = rankdata(gene_expr)
        
        # Mean rank per cluster
        mean_ranks = []
        for l in unique_labels:
            mean_ranks.append(np.mean(ranks[labels == l]))
            
        mean_ranks = np.array(mean_ranks)
        
        # Cluster with max mean rank
        max_rank = np.max(mean_ranks)
        max_indices = np.where(mean_ranks == max_rank)[0]
        
        if len(max_indices) > 1:
            auroc_arr[i] = np.nan
            clusts_arr[i] = np.nan
            p_values[i] = 1.0
            continue
            
        pos_idx = max_indices[0]
        pos_label = unique_labels[pos_idx]
        
        # Binary truth: 1 if in pos_label, 0 otherwise
        truth = (labels == pos_label).astype(int)
        
        # AUROC of ranks vs truth
        try:
            if len(np.unique(truth)) < 2:
                 auroc_arr[i] = 0.5
            else:
                auc = roc_auc_score(truth, ranks)
                auroc_arr[i] = auc
        except ValueError:
            auroc_arr[i] = 0.5
            
        clusts_arr[i] = pos_label
        
        # P-value (Mann-Whitney U / Wilcoxon Rank Sum)
        pos_scores = ranks[truth == 1]
        neg_scores = ranks[truth == 0]
        
        try:
            # alternative='two-sided' matches R wilcox.test default
            _, p = mannwhitneyu(pos_scores, neg_scores, alternative='two-sided')
            p_values[i] = p
        except ValueError:
             p_values[i] = 1.0
             
    # Adjust p-values
    p_values = p_adjust_holm(p_values)
    
    return {
        'auroc': auroc_arr,
        'clusts': clusts_arr,
        'pvalue': p_values
    }

def get_outl_cells(data, labels):
    """
    Find outlier cells (Robust PCA + MCD).
    
    Returns
    -------
    np.ndarray
        (n_cells,) Outlier scores.
    """
    n_cells = data.shape[0]
    outlier_scores = np.zeros(n_cells)
    unique_labels = np.unique(labels)
    
    chisq_quantile = 0.9999
    
    for l in unique_labels:
        mask = (labels == l)
        cluster_data = data[mask, :] # (n_cells_in_cluster, n_genes)
        
        n_cluster_cells = cluster_data.shape[0]
        
        if n_cluster_cells <= 6:
            # Too small matches R logic
            continue
            
        # Robust PCA (Approximation using standard PCA for dimensionality reduction)
        # R uses PcaHubert.
        # We use standard PCA then MinCovDet.
        # Determine components: R uses min(3, auto).
        # We use min(3, n_samples, n_features)
        n_comp = min(3, n_cluster_cells, cluster_data.shape[1])
        if n_comp < 1:
            continue
            
        try:
            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(cluster_data) # (n_cluster_cells, n_comp)
            
            # MCD
            mcd = MinCovDet()
            mcd.fit(scores)
            
            # Mahalanobis distance (squared)
            mah_dist = mcd.mahalanobis(scores)
            
            # Score = sqrt(mah) - sqrt(chisq)
            # df = n_comp (number of variables in MCD)
            threshold = np.sqrt(chi2.ppf(chisq_quantile, df=n_comp))
            outliers = np.sqrt(mah_dist) - threshold
            outliers[outliers < 0] = 0
            
            outlier_scores[mask] = outliers
            
        except Exception as e:
            # Warning can be added here if needed
            pass
            
    return outlier_scores
