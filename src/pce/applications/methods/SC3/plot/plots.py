import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker, cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def _ensure_dir(file_path):
    """
    Helper to ensure directory exists for a given file path.

    Parameters
    ----------
    file_path : str
        The full path to the file. The directory part of this path will be checked
        and created if it does not exist.
    """
    if file_path:
        out_dir = os.path.dirname(file_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)


def _save_fig(file_path, g):
    """
    Helper to save figure to file.

    Parameters
    ----------
    file_path : str
        The base path to save the figure (without extension or with).
        The function will generate .png, .pdf, and .svg files.
    g : matplotlib.figure.Figure or seaborn.matrix.ClusterGrid
        The figure object to save.

    Returns
    -------
    tuple
        A tuple containing paths to the saved PNG, PDF, and SVG files.
        (png_path, pdf_path, svg_path)
    """
    if file_path:
        _ensure_dir(file_path)

        # Split file name and extension
        base_name, ext = os.path.splitext(file_path)
        png_path = f"{base_name}.png"
        pdf_path = f"{base_name}.pdf"
        svg_path = f"{base_name}.svg"

        # Save as PNG, PDF, SVG
        g.savefig(png_path, dpi=300, bbox_inches='tight')
        g.savefig(pdf_path, dpi=300, bbox_inches='tight')
        g.savefig(svg_path, dpi=300, bbox_inches='tight')

        return png_path, pdf_path, svg_path

    else:
        raise Exception("Please specify a valid file path.")


def plot_consensus(consensus_matrix, labels=None, show_labels=False, file_path=None):
    """
    Plot consensus matrix matching R SC3's visual style exactly.

    Refinements include multi-step gradient color logic, delicate grid borders,
    and physical separation gaps between clusters.

    Parameters
    ----------
    consensus_matrix : np.ndarray
        A square consensus matrix where values range from 0 to 1.
    labels : np.ndarray or list, optional
        Cluster labels for each sample. Used to draw gaps between clusters.
        If None, gaps are not drawn.
    show_labels : bool, optional
        Whether to show labels on the axes. Default is False.
    file_path : str, optional
        Path to save the plot. If provided, saves as PNG, PDF, and SVG.

    Returns
    -------
    None
    """
    if consensus_matrix is None:
        return

    # --- 1. Clustering algorithm (R: Euclidean + Complete) ---
    dist_vector = pdist(consensus_matrix, metric='euclidean')
    row_linkage = linkage(dist_vector, method='complete')
    col_linkage = row_linkage

    # --- 2. R's 7-segment color logic ---
    # Build hybrid color palette
    custom_colors = ['#4575B4', '#91BFDB', '#E0F3F8', '#FFFFBF', '#FEE090', '#FC8D59', '#D73027']
    custom_cmap = LinearSegmentedColormap.from_list("sc3_imitation", custom_colors)

    # --- 4. Visual parameters ---
    n_cells = consensus_matrix.shape[0]

    # Grid line settings (mimic R's small squares)
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
        dendrogram_ratio=0.1,
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

    # --- 5. Draw partition lines (Gaps) ---
    if labels is not None:
        reordered_ind = g.dendrogram_row.reordered_ind
        reordered_labels = np.array(labels)[reordered_ind]
        boundaries = np.where(reordered_labels[:-1] != reordered_labels[1:])[0] + 1

        # Draw bold white lines to mimic R's cutree effect
        gap_lw = 3
        g.ax_heatmap.hlines(boundaries, *g.ax_heatmap.get_xlim(), color='white', linewidth=gap_lw, clip_on=True, zorder=10)
        g.ax_heatmap.vlines(boundaries, *g.ax_heatmap.get_ylim(), color='white', linewidth=gap_lw, clip_on=True, zorder=10)

    # --- 6. Strictly align Colorbar to the top ---
    heatmap_pos = g.ax_heatmap.get_position()

    # Set Colorbar dimensions
    cb_width = 0.02
    cb_height = 0.30

    # Calculate position:
    cb_left = heatmap_pos.x1 + 0.02
    cb_bottom = heatmap_pos.y1 - cb_height

    # Apply new position
    g.cax.set_position([cb_left, cb_bottom, cb_width, cb_height])

    # --- 6. Detailed refinement ---
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(left=False, bottom=False)

    # Save figure
    png_path, pdf_path, svg_path = _save_fig(file_path, g)
    print(f"\nConsensus plot saved to {os.path.dirname(file_path)}")
    print(f"- File name: {os.path.basename(png_path)}")
    print(f"- File name: {os.path.basename(pdf_path)}")
    print(f"- File name: {os.path.basename(svg_path)}")


def plot_silhouette(consensus_matrix, labels, file_path=None):
    """
    Plot silhouette indexes of the cells.

    Refactored to match R's 'cluster::plot.silhouette' visual style exactly:
    - Horizontal bars (black).
    - Grouped by cluster with gaps.
    - Right-side statistics (Cluster : Size | Avg Score).

    Parameters
    ----------
    consensus_matrix : np.ndarray
        Pairwise distance or similarity matrix used to calculate silhouette scores.
        Note: The function assumes this is a similarity matrix and converts it to
        distance using `pdist(..., metric='euclidean')`.
    labels : np.ndarray or list
        Cluster labels for each cell.
    file_path : str, optional
        Path to save the plot. If provided, saves as PNG, PDF, and SVG.

    Returns
    -------
    None
    """
    if consensus_matrix is None or labels is None:
        print("Consensus matrix or labels missing.")
        return

    n_cells = len(labels)

    # --- 1. Calculate Silhouette Scores ---
    dist_mat = squareform(pdist(consensus_matrix, metric='euclidean'))
    silhouette_vals = silhouette_samples(dist_mat, labels, metric='precomputed')
    global_avg_score = np.mean(silhouette_vals)

    # --- 2. Data organization and sorting ---
    df = pd.DataFrame({'label': labels, 'score': silhouette_vals})
    # Calculate statistics for each Cluster: count (n) and average score (ave)
    cluster_stats = df.groupby('label')['score'].agg(['count', 'mean']).sort_index()

    # Sorting logic:
    # Primary: Cluster ID (ascending)
    # Secondary: Score (descending)
    df = df.sort_values(by=['label', 'score'], ascending=[True, False])

    # --- 3. Plotting coordinate calculation (Core replication logic) ---
    y_positions = []
    current_y = 0
    gap = 2
    cluster_label_pos = {}
    unique_labels = sorted(df['label'].unique())

    # Iterate through each Cluster to generate coordinates
    for i, clust in enumerate(unique_labels):
        clust_data = df[df['label'] == clust]
        n_items = len(clust_data)
        start_y = current_y

        # Generate Y coordinates for this group (0.5, 1.5, 2.5...)
        y_pos_group = np.arange(start_y, start_y + n_items) + 0.5
        y_positions.extend(y_pos_group)

        # Record the center position of the group for text placement
        cluster_center = start_y + (n_items / 2.0)

        # Format right-side text: "j : n_j | ave_i s_i"
        avg_s = cluster_stats.loc[clust, 'mean']
        count_s = cluster_stats.loc[clust, 'count']
        display_clust = int(clust) + 1
        stats_text = f"{display_clust} : {int(count_s)} | {avg_s:.2f}"
        cluster_label_pos[cluster_center] = stats_text
        current_y += n_items + gap

    # --- 4. Plotting ---
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Draw horizontal bar chart, all black
    ax.barh(y_positions, df['score'].values, height=0.8, color='black', edgecolor='black', linewidth=0)

    # --- 5. Style replication ---
    # Invert Y axis so Cluster 1 (near y=0) is at the top
    ax.set_ylim(current_y - gap + 1, 0)

    # Set X axis range
    ax.set_xlim(0, 1.05)

    # Remove Y axis ticks
    ax.set_yticks([])

    # Add statistics text on the right (Cluster : Count | Avg)
    for y, text in cluster_label_pos.items():
        ax.text(1.02, y, text, ha='left', va='center', fontsize=10, color='black')

    # Add top-right Header
    header_y_pos = -2
    formula_str = r"$\mathsf{j} : \mathsf{n}_{\mathsf{j}} \mid \mathsf{ave}_{\mathsf{i} \in \mathsf{C}_{\mathsf{j}}} \ \mathsf{s}_{\mathsf{i}}$"
    ax.text(1.02, header_y_pos, formula_str, ha='left', va='bottom', fontsize=10)
    cluster_str = f"{len(unique_labels)} clusters $\mathsf{{C}}_{{\mathsf{{j}}}}$"
    ax.text(1.02, header_y_pos - gap*2, cluster_str, ha='left', va='bottom', fontsize=10)

    # Top-left info "n = 90"
    ax.text(0, header_y_pos, f"n = {n_cells}", ha='left', va='bottom', fontsize=10, fontweight='normal')

    # R Title: "Silhouette plot of (x = clusts, dist = diss)"
    ax.set_title("Silhouette plot of (x = clusters, dist = diss)", loc='left', pad=30, fontsize=12, fontweight='bold')

    # X axis label
    xlabel_str = r"Silhouette width $\mathsf{s}_{\mathsf{i}}$"
    ax.set_xlabel(xlabel_str, fontsize=10)

    # R: "Average silhouette width : 0.92"
    plt.figtext(0.125, 0.02, f"Average silhouette width : {global_avg_score:.2f}", fontsize=10)

    # Remove top and right spines, keep left and bottom
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Adjust layout to show right-side text
    plt.subplots_adjust(right=0.8, bottom=0.1)

    # Save figure
    if file_path:
        _ensure_dir(file_path)
        base_name, ext = os.path.splitext(file_path)
        png_path = f"{base_name}.png"
        pdf_path = f"{base_name}.pdf"
        svg_path = f"{base_name}.svg"
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        plt.savefig(svg_path)
        print(f"\nSilhouette plot saved to {os.path.dirname(file_path)}")
        print(f"- File name: {os.path.basename(png_path)}")
        print(f"- File name: {os.path.basename(pdf_path)}")
        print(f"- File name: {os.path.basename(svg_path)}")


def plot_expression(data, labels, consensus_matrix, seed=2026, file_path=None):
    """
    Plot expression matrix matching R SC3's visual style exactly.

    Logic aligned with R 'sc3_plot_expression':
    1. If genes > 100, perform K-Means (k=100) on genes to reduce rows.
    2. Columns (cells) are ordered by the consensus matrix clustering (hc).
    3. Use the specific SC3 7-color palette.

    Parameters
    ----------
    data : np.ndarray
        Expression matrix.
    labels : np.ndarray or list
        Cluster labels for each cell.
    consensus_matrix : np.ndarray
        Consensus matrix used for column clustering linkage.
    seed : int, optional
        Random seed for K-Means if gene reduction is performed. Default is 2026.
    file_path : str, optional
        Path to save the plot. If provided, saves as PNG, PDF, and SVG.

    Returns
    -------
    None
    """
    if data is None or labels is None or consensus_matrix is None:
        print("Data, labels, or consensus_matrix missing.")
        return

    # --- 1. Data shape handling ---
    if data.shape[0] == len(labels):
        plot_data = data.T
    else:
        plot_data = data

    n_genes, n_cells = plot_data.shape

    # --- 2. Core logic from R: Gene dimensionality reduction (K-Means) ---
    if n_genes > 100:
        kmeans = KMeans(n_clusters=100, random_state=seed, n_init=10)
        kmeans.fit(plot_data)
        plot_data = kmeans.cluster_centers_

    # --- 3. Visual configuration (Mimic R) ---
    # 3.1 Custom color palette
    sc3_colors = ['#4575B4', '#91BFDB', '#E0F3F8', '#FFFFBF', '#FEE090', '#FC8D59', '#D73027']
    sc3_cmap = LinearSegmentedColormap.from_list("sc3_expression", sc3_colors)

    # 3.2 Calculate column clustering (Use Consensus Matrix to ensure consistency)
    dist_vector = pdist(consensus_matrix, metric='euclidean')
    col_linkage = linkage(dist_vector, method='complete')

    # 3.3 Grid line width
    lw = 0.05
    if n_cells < 100:
        lw = 0.5
    elif n_cells < 500:
        lw = 0.1

    linecolor = '#808080'

    # --- 4. Plotting (Clustermap) ---
    plt.figure(figsize=(10, 10))

    g = sns.clustermap(
        plot_data,
        cmap=sc3_cmap,
        col_linkage=col_linkage,
        row_cluster=True,
        dendrogram_ratio=(0.1, 0.1),
        linewidths=lw,
        linecolor=linecolor,
        xticklabels=False,
        yticklabels=False,
        cbar_pos=(1.02, 0.70, 0.02, 0.20),
        tree_kws={'linewidths': 1.0}
    )

    # --- 5. Dynamic Colorbar alignment (Core modification) ---
    heatmap_pos = g.ax_heatmap.get_position()

    # Set Colorbar dimensions
    cb_width = 0.02
    cb_height = 0.20
    cb_gap = 0.02

    # Calculate new position:
    cb_left = heatmap_pos.x1 + cb_gap
    cb_bottom = heatmap_pos.y1 - cb_height
    g.cax.set_position([cb_left, cb_bottom, cb_width, cb_height])

    # --- 6. Final refinement (Gaps & Title) ---
    # Draw column partition lines (Gaps)
    reordered_col_ind = g.dendrogram_col.reordered_ind
    reordered_labels = np.array(labels)[reordered_col_ind]
    boundaries = np.where(reordered_labels[:-1] != reordered_labels[1:])[0] + 1
    g.ax_heatmap.vlines(boundaries, *g.ax_heatmap.get_ylim(), color='white', linewidth=3)
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    # Adjust Colorbar style
    g.cax.tick_params(labelsize=10, axis='y', length=0)

    # Save figure
    png_path, pdf_path, svg_path = _save_fig(file_path, g)
    print(f"\nExpression plot saved to {os.path.dirname(file_path)}")
    print(f"- File name: {os.path.basename(png_path)}")
    print(f"- File name: {os.path.basename(pdf_path)}")
    print(f"- File name: {os.path.basename(svg_path)}")


def plot_de_genes(data, labels, de_results_df, consensus_matrix, p_val=0.01, file_path=None):
    """
    Plot expression of DE genes using the DE results DataFrame directly.

    Matches sc3_plot_de_genes in R.

    Parameters
    ----------
    data : np.ndarray
        Expression matrix (n_cells, n_genes).
        **Important**: The columns of this matrix must match the rows of de_results_df 1-to-1.
        (i.e., data.shape[1] == de_results_df.shape[0])
    labels : np.ndarray or list
        Cluster labels for each cell.
    de_results_df : pd.DataFrame
        DataFrame with shape (n_genes, 2).
        - Col 0: Gene Names (e.g., 'feature_symbol')
        - Col 1: P-values (e.g., 'sc3_k_de_padj')
    consensus_matrix : np.ndarray
        Consensus matrix used for column clustering linkage.
    p_val : float, optional
        Significance threshold. Default is 0.01.
    file_path : str, optional
        Path to save the plot. If provided, saves as PNG, PDF, and SVG.

    Returns
    -------
    None
    """

    # --- 1. Parse DataFrame ---
    if not isinstance(de_results_df, pd.DataFrame):
        print("Error: de_results_df must be a pandas DataFrame.")
        return

    if data.shape[1] != de_results_df.shape[0]:
        print(f"Error: Data columns ({data.shape[1]}) do not match DE results rows ({de_results_df.shape[0]}).")
        return

    # Extract gene names (col 1) and P-values (col 2)
    all_gene_names = de_results_df.iloc[:, 0].values.astype(str)
    all_p_values = de_results_df.iloc[:, 1].values

    # --- 2. Data cleaning (Handle NaN) ---
    all_p_values = np.array(all_p_values, dtype=float)
    all_p_values[np.isnan(all_p_values)] = 1.0

    # --- 3. Filter Top 50 DE Genes ---
    sig_indices = np.where(all_p_values < p_val)[0]

    if len(sig_indices) == 0:
        print(f"No DE genes found with p-value < {p_val}.")
        return

    # Sort significant genes by p-value (ascending)
    sig_pvals = all_p_values[sig_indices]
    sorted_local_indices = np.argsort(sig_pvals)
    # Map back to global indices (these correspond to data columns and gene_names positions)
    sorted_global_indices = sig_indices[sorted_local_indices]

    # Take top 50
    top_50_indices = sorted_global_indices[:50]
    top_50_pvals = all_p_values[top_50_indices]
    top_50_names = all_gene_names[top_50_indices]

    # --- 4. Extract plotting data ---
    subset_data = data[:, top_50_indices].T
    df_plot = pd.DataFrame(subset_data, index=top_50_names)

    # --- 5. Visual configuration (R style replication) ---
    custom_colors = ['#4575B4', '#91BFDB', '#E0F3F8', '#FFFFBF', '#FEE090', '#FC8D59', '#D73027']
    sc3_cmap = LinearSegmentedColormap.from_list("sc3_imitation", custom_colors)
    lighter_greens = ['#EDF8FB', '#B2E2E2', '#66C2A4', '#238B45']
    smooth_greens = LinearSegmentedColormap.from_list("GreensSmooth", lighter_greens, N=256)
    discrete_greens = LinearSegmentedColormap.from_list("Greens4", lighter_greens, N=4)

    # Row annotation (log10_padj) - Left green bar
    safe_pvals = top_50_pvals.copy()
    safe_pvals[safe_pvals < 1e-17] = 1e-17
    log10_padj = -np.log10(safe_pvals)

    # Normalization
    norm = mcolors.Normalize(vmin=log10_padj.min(), vmax=log10_padj.max())

    # Use gene names as index to ensure automatic alignment with heatmap
    row_colors = pd.Series(
        log10_padj, index=top_50_names
    ).map(lambda x: mcolors.to_hex(smooth_greens(norm(x))))
    row_colors.name = "log10_padj"

    # --- Calculate Consensus Clustering Linkage ---
    dist_vector = pdist(consensus_matrix, metric='euclidean')
    col_linkage_obj = linkage(dist_vector, method='complete')

    # --- 6. Plotting (Clustermap) ---
    plt.figure(figsize=(10, 10))
    fontsize = 10 if len(top_50_names) <= 30 else 8

    g = sns.clustermap(
        df_plot,
        cmap=sc3_cmap,
        row_cluster=False,
        col_cluster=True,
        col_linkage=col_linkage_obj,
        # col_colors=col_colors.to_numpy(),
        row_colors=row_colors,
        dendrogram_ratio=(0.04, 0.10),
        xticklabels=False,
        yticklabels=True,
        cbar_pos=(1.02, 0.60, 0.02, 0.30),
        tree_kws={'linewidths': 1.5},
        # cbar_kws={'label': 'Expression'}
    )

    # --- 7. Manually force green bar position and width ---
    heatmap_pos = g.ax_heatmap.get_position()

    # Calculate new position for green bar
    rc_width = 0.02
    rc_gap = 0.005
    rc_new_pos = [heatmap_pos.x0 - rc_width - rc_gap, heatmap_pos.y0, rc_width, heatmap_pos.height]
    g.ax_row_colors.set_position(rc_new_pos)
    g.ax_row_colors.set_xticks([])
    g.ax_row_colors.set_xlabel("log10_padj", fontsize=fontsize, fontweight='bold', rotation=-90)

    # --- 8. Dynamic Colorbar alignment ---
    cb_width = 0.02
    cb_height = 0.20
    cb_bottom = heatmap_pos.y1 - cb_height
    g.cax.set_position([1.002, cb_bottom, cb_width, cb_height])

    # --- Add log10_padj legend ---
    # # log10_padj legend below Colorbar
    # gap = 0.05
    # padj_cb_height = 0.08
    # padj_cb_bottom = cb_bottom - gap - padj_cb_height
    # cax_padj = g.figure.add_axes([1.002, padj_cb_bottom, cb_width, padj_cb_height])
    # mappable_green = cm.ScalarMappable(norm=norm, cmap=discrete_greens)
    # cb_padj = plt.colorbar(mappable_green, cax=cax_padj, orientation='vertical')
    # cax_padj.set_title("log10_padj", fontsize=fontsize, loc='left', pad=8, fontweight='bold')
    # cax_padj.tick_params(labelsize=fontsize, length=0)
    # cb_padj.outline.set_visible(False)
    # min_val = log10_padj.min()
    # max_val = log10_padj.max()
    # cax_padj.set_yticks([min_val, max_val])
    # cax_padj.set_yticklabels([f"{min_val:.1f}", f"{max_val:.1f}"])

    # log10_padj legend to the right of Colorbar
    gap = 0.05
    padj_cb_left = 1.002 + cb_width + gap
    padj_cb_height = 0.08
    main_cb_top = cb_bottom + cb_height
    safe_text_gap = 0.02
    padj_cb_bottom = main_cb_top - safe_text_gap - padj_cb_height
    cax_padj = g.figure.add_axes([padj_cb_left, padj_cb_bottom, cb_width, padj_cb_height])
    mappable_green = cm.ScalarMappable(norm=norm, cmap=discrete_greens)
    cb_padj = plt.colorbar(mappable_green, cax=cax_padj, orientation='vertical')
    g.figure.text(
        padj_cb_left,
        main_cb_top,
        "log10_padj",
        fontsize=fontsize,
        fontweight='bold',
        ha='left',
        va='top'
    )
    cax_padj.tick_params(labelsize=fontsize, length=0)
    cb_padj.outline.set_visible(False)
    min_val = log10_padj.min()
    max_val = log10_padj.max()
    cax_padj.set_yticks([min_val, max_val])
    cax_padj.set_yticklabels([f"{min_val:.1f}", f"{max_val:.1f}"])

    # --- 9. Detailed refinement ---
    # Colorbar tick settings
    g.cax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    g.cax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    g.cax.tick_params(
        labelsize=fontsize,
        length=0
    )

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    # Adjust right-side gene name font
    g.ax_heatmap.tick_params(
        axis='y',
        labelright=True,
        labelleft=False,
        rotation=0,
        length=0
    )

    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fontsize)

    # Draw column partition lines (Gaps)
    if hasattr(g.dendrogram_col, 'reordered_ind'):
        reordered_col_ind = g.dendrogram_col.reordered_ind
        reordered_labels = np.array(labels)[reordered_col_ind]
        boundaries = np.where(reordered_labels[:-1] != reordered_labels[1:])[0] + 1
        g.ax_heatmap.vlines(boundaries, *g.ax_heatmap.get_ylim(), color='white', linewidth=3)

    # Save figure
    png_path, pdf_path, svg_path = _save_fig(file_path, g)
    print(f"\nDE genes plot saved to {os.path.dirname(file_path)}")
    print(f"- File name: {os.path.basename(png_path)}")
    print(f"- File name: {os.path.basename(pdf_path)}")
    print(f"- File name: {os.path.basename(svg_path)}")


def plot_markers(data, labels, marker_res, consensus_matrix, auroc_thr=0.85, p_val_thr=0.01, file_path=None):
    """
    Plot expression of marker genes matching R SC3 style exactly.

    Parameters
    ----------
    data : np.ndarray
        Expression matrix (n_cells, n_genes). Example: (90, 20214).
    labels : np.ndarray
        Cluster labels for each cell (n_cells,).
    marker_res : pd.DataFrame
        DataFrame containing marker statistics.
        Expected columns include 'feature_symbol' and columns ending in '_clusts', '_padj', '_auroc'.
        Example shape: (20214, 4).
    consensus_matrix : np.ndarray
        (n_cells, n_cells) matrix. Used for column clustering linkage to match SC3 behavior.
    auroc_thr : float, optional
        AUROC threshold for marker selection. Default is 0.85.
    p_val_thr : float, optional
        P-value threshold for marker selection. Default is 0.01.
    file_path : str, optional
        Path to save the plot. If provided, saves as PNG, PDF, and SVG.

    Returns
    -------
    None
    """

    # --- 1. Column name standardization and data cleaning ---
    if not isinstance(marker_res, pd.DataFrame):
        print("Error: marker_res must be a pandas DataFrame.")
        return

    # Automatically identify column names (prefixes change with k, e.g., sc3_6_markers...)
    col_map = {}
    for col in marker_res.columns:
        if col == 'feature_symbol':
            col_map['symbol'] = col
        elif col.endswith('_clusters'):
            col_map['clusts'] = col
        elif col.endswith('_padj'):
            col_map['pvalue'] = col
        elif col.endswith('_auroc'):
            col_map['auroc'] = col

    required_keys = ['symbol', 'clusts', 'pvalue', 'auroc']
    if not all(k in col_map for k in required_keys):
        print(f"Error: Could not identify required columns in marker_res. Found: {marker_res.columns}")
        return

    # Create standardized temporary DataFrame
    df_markers = marker_res.copy()
    df_markers = df_markers.rename(columns={
        col_map['symbol']: 'gene_name',
        col_map['clusts']: 'cluster',
        col_map['pvalue']: 'pvalue',
        col_map['auroc']: 'auroc'
    })

    # Record original indices (assuming marker_res row order matches data column order)
    df_markers['original_index'] = np.arange(len(df_markers))

    # --- 2. Filtering and sorting ---
    # Filter: p_val < thr AND auroc > thr
    mask = (df_markers['pvalue'] < p_val_thr) & (df_markers['auroc'] > auroc_thr)
    valid_markers = df_markers[mask].copy()

    if valid_markers.empty:
        print("No markers found with current thresholds.")
        return

    # Select Top 10 for each Cluster
    top_genes_indices = []
    top_genes_names = []
    gene_to_cluster = []

    unique_clusters = sorted(valid_markers['cluster'].unique())

    for c in unique_clusters:
        c_data = valid_markers[valid_markers['cluster'] == c]
        c_data = c_data.sort_values('auroc', ascending=False)
        top_10 = c_data.head(10)

        top_genes_indices.extend(top_10['original_index'].tolist())
        top_genes_names.extend(top_10['gene_name'].tolist())
        gene_to_cluster.extend([c] * len(top_10))

    # --- 3. Build plotting matrix ---
    # data is (Cells, Genes), we need (Genes, Cells)
    subset_data = data[:, top_genes_indices].T  # Shape: (n_selected_genes, n_cells)
    df_plot = pd.DataFrame(subset_data, index=top_genes_names)

    # --- 4. Visual configuration (Strictly mimic R) ---
    # 4.1 7-segment Heatmap color
    sc3_colors = ['#4575B4', '#91BFDB', '#E0F3F8', '#FFFFBF', '#FEE090', '#FC8D59', '#D73027']
    sc3_cmap = LinearSegmentedColormap.from_list("sc3_markers", sc3_colors)

    # 4.2 Cluster colors (Mimic markers.png palette)
    r_style_palette = ['#F37CEB', '#F98A81', '#83B0F7', '#00C753', '#00D7D9', '#CCB100']
    # If Cluster count exceeds preset, extend palette
    if len(unique_clusters) > len(r_style_palette):
        extra_colors = sns.color_palette("Set3", len(unique_clusters)).as_hex()
        cluster_colors_list = r_style_palette + extra_colors
    else:
        cluster_colors_list = r_style_palette

    # Build mapping: Cluster ID -> Color Hex
    cluster_color_map = {c: cluster_colors_list[i] for i, c in enumerate(unique_clusters)}

    # 4.3 Create left side color bar data (Row Colors)
    row_colors = pd.Series(gene_to_cluster).map(cluster_color_map).tolist()

    # 4.4 Calculate column clustering (Use Consensus Matrix)
    dist_vector = pdist(consensus_matrix, metric='euclidean')
    col_linkage_obj = linkage(dist_vector, method='complete')

    # --- 5. Plotting (Clustermap) ---
    n_genes, n_cells = df_plot.shape

    # Set physical size of each small square (inches)
    cell_unit_size = 0.15

    # Physical size of the matrix itself
    matrix_w_inches = n_cells * cell_unit_size
    matrix_h_inches = n_genes * cell_unit_size

    # Define fixed dimensions for surrounding elements (inches)
    dendrogram_h_inches = 1.2  # Top dendrogram height
    legend_w_inches = 2.0  # Right legend reserved width
    label_w_inches = 1.0  # Right gene name reserved width
    left_margin = 0.5  # Left Row Colors bar width
    bottom_margin = 0.2  # Bottom margin

    # Calculate total figsize
    total_w = left_margin + matrix_w_inches + label_w_inches + legend_w_inches
    total_h = matrix_h_inches + dendrogram_h_inches + bottom_margin

    # Calculate dendrogram_ratio (seaborn needs ratio, not absolute value)
    # We want the tree height to be dendrogram_h_inches
    # col_ratio = Tree height / Total height
    col_dendro_ratio = dendrogram_h_inches / total_h

    # Prevent ratio from being too small or too large causing errors
    col_dendro_ratio = max(0.05, min(0.3, col_dendro_ratio))

    # Determine font size
    fontsize = 12 if n_genes <= 30 else 10

    # Dynamically adjust grid line width
    lw = 0.5
    if df_plot.shape[1] > 100:
        lw = 0.1
    if df_plot.shape[1] > 500:
        lw = 0.05

    g = sns.clustermap(
        df_plot,
        figsize=(total_w, total_h),
        cmap=sc3_cmap,
        row_cluster=False,
        col_cluster=True,
        col_linkage=col_linkage_obj,
        row_colors=row_colors,
        dendrogram_ratio=(0.02, col_dendro_ratio),
        cbar_pos=(1.02, 0.75, 0.02, 0.15),
        linewidths=lw,
        linecolor='#808080',
        xticklabels=False,
        yticklabels=True,
        vmin=0,
        tree_kws={'linewidths': 1.5}
    )

    heatmap_ax = g.ax_heatmap

    # --- 6. Final refinement (Gaps, Labels, Legend) ---
    # 6.1 Draw GAP (Bold white partition lines)
    # [Row GAP]: Draw lines between genes of different Clusters
    cluster_ids = np.array(gene_to_cluster)
    row_boundaries = np.where(cluster_ids[:-1] != cluster_ids[1:])[0] + 1
    heatmap_ax.hlines(row_boundaries, *heatmap_ax.get_xlim(), color='white', linewidth=3)

    # [Col GAP]: Draw lines between cells of different Clusters
    reordered_col_ind = g.dendrogram_col.reordered_ind
    reordered_labels = np.array(labels)[reordered_col_ind]
    col_boundaries = np.where(reordered_labels[:-1] != reordered_labels[1:])[0] + 1
    heatmap_ax.vlines(col_boundaries, *heatmap_ax.get_ylim(), color='white', linewidth=3)

    # 6.2 Adjust Row Colors bar
    # Remove label added by Seaborn by default
    if g.ax_row_colors:
        g.ax_row_colors.set_xticklabels([])
        g.ax_row_colors.set_xlabel("")
        heatmap_pos = heatmap_ax.get_position()
        # Calculate relative ratio corresponding to physical width
        rc_width = cell_unit_size / total_w
        rc_gap = 0.05 / total_w
        rc_new_pos = [heatmap_pos.x0 - rc_width - rc_gap, heatmap_pos.y0, rc_width, heatmap_pos.height]
        g.ax_row_colors.set_position(rc_new_pos)

    # 6.3 Dynamic Colorbar alignment
    heatmap_pos = heatmap_ax.get_position()
    cb_width = 0.01
    cb_height = 0.25
    cb_bottom = heatmap_pos.y1 - cb_height
    g.cax.set_position([1.002, cb_bottom, cb_width, cb_height])
    g.cax.tick_params(axis='y', length=0)

    # 6.4 Add Cluster Legend
    # Since sns.clustermap's row_colors doesn't auto-generate a legend, we must add it manually
    legend_elements = [Patch(facecolor=cluster_color_map[c], edgecolor='#808080', label=f'{int(c)}')
                       for c in unique_clusters]

    # Calculate legend position: Place it to the right of Colorbar
    legend_left = 1.002 + cb_width + 0.03
    legend_bottom = cb_bottom
    legend_width = 0.05
    legend_height = cb_height

    # Add Legend Axes to the right of Colorbar
    legend_ax = g.figure.add_axes([legend_left, legend_bottom, legend_width, legend_height])

    # Draw title "Cluster"
    legend_ax.text(
        -0.03,
        1.0,
        "Cluster",
        transform=legend_ax.transAxes,
        fontsize=fontsize,
        fontweight='bold',
        ha='left',
        va='top'
    )

    # Draw legend content (Small squares)
    content_shift_down = 0.08
    legend_ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(0, 1.0 - content_shift_down),
        loc='upper left',
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        handleheight=2.0,
        labelspacing=0,
        borderpad=0,
        handletextpad=0.5,
        borderaxespad=0
    )
    legend_ax.axis('off')

    # 6.5 Axis and label adjustment
    heatmap_ax.set_xlabel("")
    heatmap_ax.set_ylabel("")

    # Place gene names on the right
    heatmap_ax.tick_params(axis='y', labelsize=9, labelright=True, labelleft=False, rotation=0, length=0)

    # Save figure
    png_path, pdf_path, svg_path = _save_fig(file_path, g)
    print(f"\nMarker genes plot saved to {os.path.dirname(file_path)}")
    print(f"- File name: {os.path.basename(png_path)}")
    print(f"- File name: {os.path.basename(pdf_path)}")
    print(f"- File name: {os.path.basename(svg_path)}")
