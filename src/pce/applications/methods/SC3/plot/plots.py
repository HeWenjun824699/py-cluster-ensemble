import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_samples, silhouette_score

# Define SC3-Nature methods-2017-like colormap (Blue -> White -> Red seems standard for consensus, or Blue->Red)
# R sc3_plot_consensus doc: "Similarity 0 (blue) ... similarity 1 (red)"
sc3_cmap = LinearSegmentedColormap.from_list("sc3_consensus", ["blue", "red"])

def _ensure_dir(file_path):
    """Helper to ensure directory exists for a given file path."""
    if file_path:
        out_dir = os.path.dirname(file_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

def plot_consensus(consensus_matrix, labels=None, show_labels=False, file_path=None):
    """
    Plot consensus matrix as a heatmap.
    Matches sc3_plot_consensus in R.
    
    Parameters
    ----------
    consensus_matrix : np.ndarray
        NxN consensus matrix.
    labels : np.ndarray, optional
        Cluster labels for annotation.
    show_labels : bool
        Whether to show cell names/indices.
    file_path : str, optional
        If provided, save plot to this path.
    """
    if consensus_matrix is None:
        print("No consensus matrix provided.")
        return

    # In R: pheatmap(..., cluster_rows=hc, cluster_cols=hc)
    # Seaborn clustermap performs hierarchical clustering automatically.
    
    # Prepare annotations if labels are provided
    col_colors = None
    if labels is not None:
        # Map labels to colors
        unique_labels = np.unique(labels)
        # Use a distinct palette
        palette = sns.color_palette("tab10", n_colors=len(unique_labels))
        lut = dict(zip(unique_labels, palette))
        col_colors = pd.Series(labels).map(lut)
        col_colors.name = "Cluster"

    # Convert to numpy to avoid index alignment issues with seaborn
    colors_array = col_colors.to_numpy() if col_colors is not None else None

    g = sns.clustermap(
        consensus_matrix,
        cmap=sc3_cmap,
        vmin=0, vmax=1,
        row_colors=colors_array,
        col_colors=colors_array,
        xticklabels=show_labels,
        yticklabels=show_labels,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.05, 0.18) # Adjust position to look like R's if possible, or default
    )
    
    g.ax_heatmap.set_title("Consensus Matrix")
    
    if file_path:
        _ensure_dir(file_path)
        g.savefig(file_path)
    
    # plt.show()

def plot_silhouette(consensus_matrix, labels, file_path=None):
    """
    Plot silhouette indexes of the cells.
    Matches sc3_plot_silhouette in R.
    """
    if consensus_matrix is None or labels is None:
        print("Consensus matrix or labels missing.")
        return
        
    # Calculate silhouette scores
    # R uses 'cluster::silhouette(clusts, diss)' where diss is distance from consensus
    # Dissimilarity = 1 - consensus? Or Euclidean on consensus?
    # R code: 
    #   tmp <- ED2(dat) ... diss <- as.dist(...)
    #   silh <- cluster::silhouette(clusts, diss)
    # So it uses Euclidean distance of the consensus matrix rows.
    
    # We can use sklearn silhouette_samples with metric='precomputed' if we have distance,
    # or just pass the consensus matrix if we treat it as features.
    # To match R exactly, we should compute Euclidean distance of consensus matrix.
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(consensus_matrix, metric='euclidean'))
    
    silhouette_vals = silhouette_samples(dist_mat, labels, metric='precomputed')
    
    # Average silhouette score
    avg_score = np.mean(silhouette_vals)
    
    # Organize for plotting
    # R's default plot(silh) shows bars sorted by cluster and then by score (descending)
    
    df = pd.DataFrame({'label': labels, 'silhouette': silhouette_vals})
    df = df.sort_values(['label', 'silhouette'], ascending=[True, False]).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot
    # We want a solid block for each cluster
    # Color by cluster
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    
    y_lower = 0
    for i, k in enumerate(unique_labels):
        k_vals = df[df['label'] == k]['silhouette'].values
        k_size = len(k_vals)
        
        x = np.arange(y_lower, y_lower + k_size)
        plt.bar(x, k_vals, width=1.0, color=palette[i], edgecolor='none', label=f'Cluster {k}')
        y_lower += k_size
        
    plt.title(f"Silhouette Plot (Avg Score: {avg_score:.2f})")
    plt.xlabel("Cells (sorted by cluster and silhouette width)")
    plt.ylabel("Silhouette Width")
    plt.axhline(avg_score, color="red", linestyle="--")
    plt.legend()
    
    if file_path:
        _ensure_dir(file_path)
        plt.savefig(file_path)
    # plt.show()

def plot_expression(data, labels, file_path=None):
    """
    Plot expression matrix used for SC3-Nature methods-2017 clustering as a heatmap.
    Matches sc3_plot_expression in R.
    
    Parameters
    ----------
    data : np.ndarray
        Expression matrix (n_cells, n_genes) or (n_genes, n_cells).
        R SC3-Nature methods-2017 uses (genes, cells). Python SC3-Nature methods-2017 usually (cells, genes).
        We will assume (cells, genes) and transpose for the heatmap (Genes x Cells).
    """
    # Transpose to (Genes, Cells) for visualization to match R style
    # if data is (cells, genes)
    if data.shape[0] == len(labels):
         plot_data = data.T
    else:
         plot_data = data
         
    # Labels match the columns (cells)
    
    # Prepare annotations
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    lut = dict(zip(unique_labels, palette))
    col_colors = pd.Series(labels).map(lut)
    col_colors.name = "Cluster"
    
    # R clusters columns (cells) based on hc, and clusters rows (genes) using kmeans if large
    # Here we use standard clustermap
    
    g = sns.clustermap(
        plot_data,
        cmap="viridis",
        col_cluster=True,
        row_cluster=True, # R uses k-means for genes if > 100, we simplify to standard hierarchical
        col_colors=col_colors.to_numpy(),
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Log2 Expression'}
    )
    
    g.ax_heatmap.set_title("Gene Expression")
    
    if file_path:
        _ensure_dir(file_path)
        g.savefig(file_path)
    # plt.show()

def plot_de_genes(data, labels, de_genes_dict, p_val=0.01, file_path=None):
    """
    Plot expression of DE genes.
    Matches sc3_plot_de_genes in R.
    
    de_genes_dict: result from get_de_genes (p-values)
    """
    # Need to identify top 50 DE genes
    # In Python implementation, get_de_genes returns p-values for all genes
    # data is (cells, genes)
    
    if len(de_genes_dict) != data.shape[1]:
        print("Mismatch between DE results and gene count.")
        return
        
    # Filter by p-value
    sig_indices = np.where(de_genes_dict < p_val)[0]
    
    if len(sig_indices) == 0:
        print("No DE genes found.")
        return
        
    # Sort by p-value and take top 50
    sig_pvals = de_genes_dict[sig_indices]
    sorted_indices = sig_indices[np.argsort(sig_pvals)]
    top_50_indices = sorted_indices[:50]
    
    # Extract data (Genes x Cells)
    subset_data = data[:, top_50_indices].T
    
    # Annotations
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    lut = dict(zip(unique_labels, palette))
    col_colors = pd.Series(labels).map(lut)
    col_colors.name = "Cluster"
    
    # Row annotation (Log10 p-adj)
    # row_colors? heatmap supports row_colors.
    # We can map p-value to a color or just show the heatmap.
    
    g = sns.clustermap(
        subset_data,
        cmap="viridis",
        col_cluster=True, # Cluster cells
        row_cluster=True, # Cluster genes
        col_colors=col_colors.to_numpy(),
        xticklabels=False,
        yticklabels=True,
        cbar_kws={'label': 'Expression'}
    )
    g.ax_heatmap.set_title("DE Genes Expression")
    
    if file_path:
        _ensure_dir(file_path)
        g.savefig(file_path)
    # plt.show()

def plot_markers(data, labels, marker_res, auroc_thr=0.85, p_val_thr=0.01, file_path=None):
    """
    Plot expression of marker genes.
    Matches sc3_plot_markers.
    
    marker_res: DataFrame/Dict from get_marker_genes
    """
    # marker_res is a DataFrame with columns: ['auroc', 'clusts', 'pvalue']
    # index should be gene index if not specified? 
    # In Python get_marker_genes returns a DataFrame aligned with genes? 
    # Let's check core_functions/biology.py.
    # get_marker_genes returns df with columns ["auroc", "clusts", "pvalue"]
    # rows correspond to genes in order.
    
    if not isinstance(marker_res, pd.DataFrame):
        marker_res = pd.DataFrame(marker_res, columns=["auroc", "clusts", "pvalue"])
    
    # Filter
    mask = (marker_res['pvalue'] < p_val_thr) & (marker_res['auroc'] > auroc_thr)
    valid_markers = marker_res[mask].copy()
    valid_markers['gene_index'] = valid_markers.index
    
    if valid_markers.empty:
        print("No markers found.")
        return
        
    # Select top 10 per cluster
    top_genes_indices = []
    
    unique_clusters = np.unique(valid_markers['clusts'])
    for c in unique_clusters:
        c_markers = valid_markers[valid_markers['clusts'] == c]
        # Sort by AUROC desc
        c_markers = c_markers.sort_values('auroc', ascending=False)
        top_10 = c_markers.head(10)['gene_index'].tolist()
        top_genes_indices.extend(top_10)
        
    # Extract data
    # Remove duplicates if any
    top_genes_indices = list(dict.fromkeys(top_genes_indices))
    
    subset_data = data[:, top_genes_indices].T
    
    # Annotations
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    lut = dict(zip(unique_labels, palette))
    col_colors = pd.Series(labels).map(lut)
    col_colors.name = "Cluster"
    
    g = sns.clustermap(
        subset_data,
        cmap="viridis",
        col_cluster=True,
        row_cluster=False, # Don't re-cluster markers, keep them grouped by cluster? 
                           # R pheatmap usually clusters unless order is fixed.
                           # R sc3_plot_markers: cluster_rows = FALSE.
        col_colors=col_colors.to_numpy(),
        xticklabels=False,
        yticklabels=True,
        cbar_kws={'label': 'Expression'}
    )
    g.ax_heatmap.set_title("Marker Genes Expression")
    
    if file_path:
        _ensure_dir(file_path)
        g.savefig(file_path)
    # plt.show()

def plot_cluster_stability(stability_indices, clusters, file_path=None):
    """
    Plot stability of the clusters.
    Matches sc3_plot_cluster_stability.
    
    stability_indices: list/array of stability scores.
    clusters: list/array of cluster IDs corresponding to the scores.
    """
    if stability_indices is None:
        print("No stability data.")
        return
        
    plt.figure(figsize=(8, 6))
    plt.bar(clusters, stability_indices, color='steelblue')
    plt.ylim(0, 1)
    plt.xlabel("Cluster")
    plt.ylabel("Stability Index")
    plt.title("Cluster Stability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if file_path:
        _ensure_dir(file_path)
        plt.savefig(file_path)
    # plt.show()
