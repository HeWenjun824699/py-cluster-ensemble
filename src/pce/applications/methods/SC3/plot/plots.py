import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker, cm
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_samples


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
    custom_colors = ['#4575B4', '#91BFDB', '#E0F3F8', '#FFFFBF', '#FEE090', '#FC8D59', '#D73027']
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

    # --- 5. 绘制分区线 (Gaps) ---
    if labels is not None:
        reordered_ind = g.dendrogram_row.reordered_ind
        reordered_labels = np.array(labels)[reordered_ind]
        boundaries = np.where(reordered_labels[:-1] != reordered_labels[1:])[0] + 1

        # 绘制加粗白线模拟 R 的 cutree 效果
        gap_lw = 3
        g.ax_heatmap.hlines(boundaries, *g.ax_heatmap.get_xlim(), color='white', linewidth=gap_lw, clip_on=True, zorder=10)
        g.ax_heatmap.vlines(boundaries, *g.ax_heatmap.get_ylim(), color='white', linewidth=gap_lw, clip_on=True, zorder=10)

    # --- 6. Colorbar 顶部严格对齐 ---
    heatmap_pos = g.ax_heatmap.get_position()

    # 设定 Colorbar 尺寸
    cb_width = 0.02
    cb_height = 0.30

    # 计算位置：
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
    labels : list
        Cluster labels for each cell.
    de_results_df : pd.DataFrame
        DataFrame with shape (n_genes, 2).
        - Col 0: Gene Names (e.g., 'feature_symbol')
        - Col 1: P-values (e.g., 'sc3_k_de_padj')
    p_val : float
        Significance threshold.
    """

    # --- 1. 解析 DataFrame ---
    if not isinstance(de_results_df, pd.DataFrame):
        print("Error: de_results_df must be a pandas DataFrame.")
        return

    if data.shape[1] != de_results_df.shape[0]:
        print(f"Error: Data columns ({data.shape[1]}) do not match DE results rows ({de_results_df.shape[0]}).")
        return

    # 提取基因名 (第1列) 和 P值 (第2列)
    all_gene_names = de_results_df.iloc[:, 0].values.astype(str)
    all_p_values = de_results_df.iloc[:, 1].values

    # --- 2. 数据清洗 (处理 NaN) ---
    all_p_values = np.array(all_p_values, dtype=float)
    all_p_values[np.isnan(all_p_values)] = 1.0

    # --- 3. 筛选 Top 50 DE Genes ---
    sig_indices = np.where(all_p_values < p_val)[0]

    if len(sig_indices) == 0:
        print(f"No DE genes found with p-value < {p_val}.")
        return

    # 在显著基因中，按 p-value 从小到大排序
    sig_pvals = all_p_values[sig_indices]
    sorted_local_indices = np.argsort(sig_pvals)
    # 映射回全局索引 (这些索引对应 data 的列 和 gene_names 的位置)
    sorted_global_indices = sig_indices[sorted_local_indices]

    # 截取前 50 个
    top_50_indices = sorted_global_indices[:50]
    top_50_pvals = all_p_values[top_50_indices]
    top_50_names = all_gene_names[top_50_indices]

    # --- 4. 提取绘图数据 ---
    subset_data = data[:, top_50_indices].T
    df_plot = pd.DataFrame(subset_data, index=top_50_names)

    # --- 5. 视觉配置 (R 风格复刻) ---
    custom_colors = ['#4575B4', '#91BFDB', '#E0F3F8', '#FFFFBF', '#FEE090', '#FC8D59', '#D73027']
    sc3_cmap = LinearSegmentedColormap.from_list("sc3_imitation", custom_colors)
    lighter_greens = ['#EDF8FB', '#B2E2E2', '#66C2A4', '#238B45']
    smooth_greens = LinearSegmentedColormap.from_list("GreensSmooth", lighter_greens, N=256)
    discrete_greens = LinearSegmentedColormap.from_list("Greens4", lighter_greens, N=4)

    # 行注释 (log10_padj) - 左侧绿色条
    safe_pvals = top_50_pvals.copy()
    safe_pvals[safe_pvals < 1e-17] = 1e-17
    log10_padj = -np.log10(safe_pvals)

    # 归一化
    norm = mcolors.Normalize(vmin=log10_padj.min(), vmax=log10_padj.max())

    # 使用基因名作为索引，确保与 heatmap 自动对齐
    row_colors = pd.Series(
        log10_padj, index=top_50_names
    ).map(lambda x: mcolors.to_hex(smooth_greens(norm(x))))
    row_colors.name = "log10_padj"

    # --- 计算共识聚类的 Linkage ---
    dist_vector = pdist(consensus_matrix, metric='euclidean')
    col_linkage_obj = linkage(dist_vector, method='complete')

    # --- 6. 绘图 (Clustermap) ---
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

    # --- 7. 手动强制设置绿条的位置和宽度 ---
    heatmap_pos = g.ax_heatmap.get_position()

    # 计算绿条的新位置
    rc_width = 0.02
    rc_gap = 0.005
    rc_new_pos = [heatmap_pos.x0 - rc_width - rc_gap, heatmap_pos.y0, rc_width, heatmap_pos.height]
    g.ax_row_colors.set_position(rc_new_pos)
    g.ax_row_colors.set_xticks([])
    g.ax_row_colors.set_xlabel("log10_padj", fontsize=fontsize, fontweight='bold', rotation=-90)

    # --- 8. 动态对齐 Colorbar ---
    cb_width = 0.02
    cb_height = 0.20
    cb_bottom = heatmap_pos.y1 - cb_height
    g.cax.set_position([1.002, cb_bottom, cb_width, cb_height])

    # --- 添加 log10_padj 图例 ---
    # # log10_padj 图例 位于 Colorbar 的下方
    # gap = 0.05
    # padj_cb_height = 0.08
    # padj_cb_bottom = cb_bottom - gap - padj_cb_height
    # cax_padj = g.figure.add_axes([1.002, padj_cb_bottom, cb_width, padj_cb_height])

    # log10_padj 图例 位于 Colorbar 的右侧
    gap = 0.05
    padj_cb_left = 1.002 + cb_width + gap
    padj_cb_height = 0.08
    padj_cb_bottom = (cb_bottom + cb_height) - padj_cb_height
    cax_padj = g.figure.add_axes([padj_cb_left, padj_cb_bottom, cb_width, padj_cb_height])

    mappable_green = cm.ScalarMappable(norm=norm, cmap=discrete_greens)
    cb_padj = plt.colorbar(mappable_green, cax=cax_padj, orientation='vertical')
    cax_padj.set_title("log10_padj", fontsize=fontsize, loc='left', pad=8, fontweight='bold')
    cax_padj.tick_params(labelsize=fontsize)
    cb_padj.outline.set_visible(False)
    cax_padj.tick_params(length=0)
    min_val = log10_padj.min()
    max_val = log10_padj.max()
    cax_padj.set_yticks([min_val, max_val])
    cax_padj.set_yticklabels([f"{min_val:.1f}", f"{max_val:.1f}"])

    # --- 9. 细节修饰 ---
    # Colorbar 刻度设置
    g.cax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    g.cax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    g.cax.tick_params(
        labelsize=fontsize,
        length=0
    )

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    # 调整右侧基因名字体
    g.ax_heatmap.tick_params(
        axis='y',
        labelright=True,
        labelleft=False,
        rotation=0,
        length=0
    )

    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fontsize)

    # 绘制列分割白线 (Gaps)
    if hasattr(g.dendrogram_col, 'reordered_ind'):
        reordered_col_ind = g.dendrogram_col.reordered_ind
        reordered_labels = np.array(labels)[reordered_col_ind]
        boundaries = np.where(reordered_labels[:-1] != reordered_labels[1:])[0] + 1
        g.ax_heatmap.vlines(boundaries, *g.ax_heatmap.get_ylim(), color='white', linewidth=3)

    if file_path:
        _ensure_dir(file_path)
        g.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"DE genes plot saved to {file_path}")

    # plt.show()


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
    """

    # --- 1. 列名标准化与数据清洗 ---
    if not isinstance(marker_res, pd.DataFrame):
        print("Error: marker_res must be a pandas DataFrame.")
        return

    # 自动识别列名 (因为 k 不同，列名前缀如 sc3_6_markers... 会变)
    col_map = {}
    for col in marker_res.columns:
        if col == 'feature_symbol':
            col_map['symbol'] = col
        elif col.endswith('_clusts'):
            col_map['clusts'] = col
        elif col.endswith('_padj'):
            col_map['pvalue'] = col
        elif col.endswith('_auroc'):
            col_map['auroc'] = col

    required_keys = ['symbol', 'clusts', 'pvalue', 'auroc']
    if not all(k in col_map for k in required_keys):
        print(f"Error: Could not identify required columns in marker_res. Found: {marker_res.columns}")
        return

    # 创建标准化的临时 DataFrame
    df_markers = marker_res.copy()
    df_markers = df_markers.rename(columns={
        col_map['symbol']: 'gene_name',
        col_map['clusts']: 'cluster',
        col_map['pvalue']: 'pvalue',
        col_map['auroc']: 'auroc'
    })

    # 记录原始索引 (假设 marker_res 的行顺序对应 data 的列顺序)
    df_markers['original_index'] = np.arange(len(df_markers))

    # --- 2. 筛选与排序 ---
    # 筛选: p_val < thr AND auroc > thr
    mask = (df_markers['pvalue'] < p_val_thr) & (df_markers['auroc'] > auroc_thr)
    valid_markers = df_markers[mask].copy()

    if valid_markers.empty:
        print("No markers found with current thresholds.")
        return

    # 选取每个 Cluster 的 Top 10
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

    # --- 3. 构建绘图矩阵 ---
    # data 是 (Cells, Genes)，我们需要 (Genes, Cells)
    subset_data = data[:, top_genes_indices].T  # Shape: (n_selected_genes, n_cells)
    df_plot = pd.DataFrame(subset_data, index=top_genes_names)

    # --- 4. 视觉配置 (严格复刻 R) ---
    # 4.1 7段式 Heatmap 颜色
    sc3_colors = ['#4575B4', '#91BFDB', '#E0F3F8', '#FFFFBF', '#FEE090', '#FC8D59', '#D73027']
    sc3_cmap = LinearSegmentedColormap.from_list("sc3_markers", sc3_colors)

    # 4.2 Cluster 颜色 (复刻 markers.png 中的色盘)
    r_style_palette = ['#F37CEB', '#F98A81', '#83B0F7', '#00C753', '#00D7D9', '#CCB100']
    # 如果 Cluster 数量超过预设，扩展色盘
    if len(unique_clusters) > len(r_style_palette):
        extra_colors = sns.color_palette("Set3", len(unique_clusters)).as_hex()
        cluster_colors_list = r_style_palette + extra_colors
    else:
        cluster_colors_list = r_style_palette

    # 建立映射: Cluster ID -> Color Hex
    cluster_color_map = {c: cluster_colors_list[i] for i, c in enumerate(unique_clusters)}

    # 4.3 制作左侧颜色条数据 (Row Colors)
    row_colors = pd.Series(gene_to_cluster).map(cluster_color_map).tolist()

    # 4.4 计算列聚类 (使用 Consensus Matrix)
    dist_vector = pdist(consensus_matrix, metric='euclidean')
    col_linkage_obj = linkage(dist_vector, method='complete')

    # --- 5. 绘图 (Clustermap) ---
    n_genes, n_cells = df_plot.shape

    # 设定每个小方格的物理尺寸 (英寸)
    cell_unit_size = 0.15

    # 矩阵本身的物理尺寸
    matrix_w_inches = n_cells * cell_unit_size
    matrix_h_inches = n_genes * cell_unit_size

    # 定义周边元素的固定尺寸 (英寸)
    dendrogram_h_inches = 1.2  # 顶部树状图高度
    legend_w_inches = 2.0  # 右侧图例预留宽度
    label_w_inches = 1.0  # 右侧基因名预留宽度
    left_margin = 0.5  # 左侧 Row Colors 条宽度
    bottom_margin = 0.2  # 底部留白

    # 计算总 figsize
    total_w = left_margin + matrix_w_inches + label_w_inches + legend_w_inches
    total_h = matrix_h_inches + dendrogram_h_inches + bottom_margin

    # 计算 dendrogram_ratio (seaborn 需要的是比例，不是绝对值)
    # 我们希望树的高度是 dendrogram_h_inches
    # col_ratio = 树的高度 / 总高度
    col_dendro_ratio = dendrogram_h_inches / total_h

    # 防止比例过小或过大导致报错
    col_dendro_ratio = max(0.05, min(0.3, col_dendro_ratio))

    # 确定字体大小
    fontsize = 12 if n_genes <= 30 else 10

    # 动态调整网格线宽
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

    # --- 6. 后期精修 (Gaps, Labels, Legend) ---
    # 6.1 绘制 GAP (白色粗分割线)
    # [行 GAP]：在不同 Cluster 的基因之间画线
    cluster_ids = np.array(gene_to_cluster)
    row_boundaries = np.where(cluster_ids[:-1] != cluster_ids[1:])[0] + 1
    heatmap_ax.hlines(row_boundaries, *heatmap_ax.get_xlim(), color='white', linewidth=3)

    # [列 GAP]：在不同 Cluster 的细胞之间画线
    reordered_col_ind = g.dendrogram_col.reordered_ind
    reordered_labels = np.array(labels)[reordered_col_ind]
    col_boundaries = np.where(reordered_labels[:-1] != reordered_labels[1:])[0] + 1
    heatmap_ax.vlines(col_boundaries, *heatmap_ax.get_ylim(), color='white', linewidth=3)

    # 6.2 调整 Row Colors 条
    # 去掉 Seaborn 默认添加的 label
    if g.ax_row_colors:
        g.ax_row_colors.set_xticklabels([])
        g.ax_row_colors.set_xlabel("")
        heatmap_pos = heatmap_ax.get_position()
        # 计算物理宽度对应的相对比例
        rc_width = cell_unit_size / total_w
        rc_gap = 0.05 / total_w
        rc_new_pos = [heatmap_pos.x0 - rc_width - rc_gap, heatmap_pos.y0, rc_width, heatmap_pos.height]
        g.ax_row_colors.set_position(rc_new_pos)

    # 6.3 动态对齐 Colorbar
    heatmap_pos = heatmap_ax.get_position()
    cb_width = 0.01
    cb_height = 0.25
    cb_bottom = heatmap_pos.y1 - cb_height
    g.cax.set_position([1.002, cb_bottom, cb_width, cb_height])

    # 6.4 添加 Cluster 图例 (Legend)
    # 因为 sns.clustermap 的 row_colors 不会自动生成图例，需手动添加
    legend_elements = [Patch(facecolor=cluster_color_map[c], edgecolor='#808080', label=f'{int(c)}')
                       for c in unique_clusters]

    # [修改] 计算图例位置：使其位于 Colorbar 右侧
    legend_left = 1.002 + cb_width + 0.03
    legend_bottom = cb_bottom
    legend_width = 0.05
    legend_height = cb_height

    # 在 Colorbar 右侧添加图例 Axes
    legend_ax = g.figure.add_axes([legend_left, legend_bottom, legend_width, legend_height])

    legend_ax.legend(
        handles=legend_elements,
        loc='upper left',
        title="Cluster",
        title_fontproperties={'weight': 'bold', 'size': fontsize},
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

    # 6.5 坐标轴与标签调整
    heatmap_ax.set_xlabel("")
    heatmap_ax.set_ylabel("")

    # 将基因名放在右侧
    heatmap_ax.tick_params(axis='y', labelsize=9, labelright=True, labelleft=False, rotation=0, length=0)

    # 保存
    if file_path:
        _ensure_dir(file_path)
        g.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Marker genes plot saved to {file_path}")

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
