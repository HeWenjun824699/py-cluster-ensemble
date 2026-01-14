import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_samples, silhouette_score


def _ensure_dir(file_path):
    """Helper to ensure directory exists for a given file path."""
    if file_path:
        out_dir = os.path.dirname(file_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)


def plot_consensus(consensus_matrix, labels=None, show_labels=False, file_path=None):
    """
    Plot consensus matrix matching R SC3's visual style exactly.

    Refinements:
    1. Color Logic: Multi-step gradient mimicking R's 'RdYlBu' (7 steps),
       but anchoring the deep blue to 'Cluster0 Blue' (#1f77b4).
    2. Grid: Delicate white borders (0.5px).
    3. Gaps: Physical separation between clusters.
    """
    if consensus_matrix is None:
        return

    # --- 1. 聚类算法 (R: Euclidean + Complete) ---
    dist_vector = pdist(consensus_matrix, metric='euclidean')
    row_linkage = linkage(dist_vector, method='complete')
    col_linkage = row_linkage

    # --- 2. R 的 7 段式变色逻辑 ---
    # 构建混合色板
    custom_colors = [
        '#4575B4',  # 0.0 (最深蓝)
        '#91BFDB',  # 0.16 (浅蓝)
        '#E0F3F8',  # 0.33 (极浅蓝)
        '#FFFFBF',  # 0.50 (米黄/白 )
        '#FEE090',  # 0.66 (浅橙)
        '#FC8D59',  # 0.83 (橙)
        '#D73027'   # 1.0 (深红)
    ]

    custom_cmap = LinearSegmentedColormap.from_list("sc3_imitation", custom_colors)

    # --- 4. 视觉参数 ---
    n_cells = consensus_matrix.shape[0]

    # 网格线设置 (模拟 R 的小方格)
    lw = 0.35
    if n_cells > 200:
        lw = 0.1
    if n_cells > 500:
        lw = 0.05
    linecolor = '#808080'

    plt.figure(figsize=(10, 9))
    df_cons = pd.DataFrame(consensus_matrix)

    g = sns.clustermap(
        df_cons,
        row_linkage=row_linkage,
        col_linkage=col_linkage,

        # 压缩树状图高度
        dendrogram_ratio=0.1,

        # 配色与样式
        cmap=custom_cmap,
        vmin=0, vmax=1,
        linewidths=lw,
        linecolor=linecolor,

        xticklabels=show_labels,
        yticklabels=show_labels,
        cbar_pos=(1.02, 0.60, 0.02, 0.30),
        # cbar_kws={'label': 'Similarity'},
        tree_kws={'linewidths': 1.0}
    )

    # --- 5. 绘制分区线 (Gaps) ---
    if labels is not None:
        reordered_ind = g.dendrogram_row.reordered_ind
        reordered_labels = np.array(labels)[reordered_ind]
        boundaries = np.where(reordered_labels[:-1] != reordered_labels[1:])[0] + 1

        # 绘制加粗白线模拟 R 的 cutree 效果
        # 这里线宽设为 3，确保能看出来是“分区”
        gap_lw = 3
        g.ax_heatmap.hlines(boundaries, *g.ax_heatmap.get_xlim(), color='white', linewidth=gap_lw, clip_on=True, zorder=10)
        g.ax_heatmap.vlines(boundaries, *g.ax_heatmap.get_ylim(), color='white', linewidth=gap_lw, clip_on=True, zorder=10)

    # --- 6. [核心修正] Colorbar 顶部严格对齐 ---
    # 获取热图绘制后的准确坐标 (Bbox)
    # heatmap_pos.y0 = 底部, heatmap_pos.y1 = 顶部
    heatmap_pos = g.ax_heatmap.get_position()

    # 设定 Colorbar 尺寸
    cb_width = 0.02
    cb_height = 0.30

    # 计算位置：
    # left = 热图右边缘 + 间距
    # bottom = 热图顶部 (y1) - Colorbar高度 (height) -> 这样顶部就齐平了
    cb_left = heatmap_pos.x1 + 0.02
    cb_bottom = heatmap_pos.y1 - cb_height

    # 应用新位置
    g.cax.set_position([cb_left, cb_bottom, cb_width, cb_height])

    # --- 6. 细节修饰 ---
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(left=False, bottom=False)

    if file_path:
        _ensure_dir(file_path)
        g.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Consensus plot saved to {file_path}")

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
