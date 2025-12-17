from typing import Optional, List, Dict, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from .utils import set_paper_style, save_fig


def plot_2d_scatter(
        X: np.ndarray,
        labels: np.ndarray,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        method: Optional[str] = 'tsne',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        **kwargs: Any
) -> None:
    """
    绘制 2D 散点图

    Args:
        X: 特征矩阵，通常是 (n_samples, n_features) 的浮点数数组
        labels: 标签向量，通常是 (n_samples,) 的整数数组
        method: 降维方法，字符串，默认 'tsne'
        title: 图表标题，可选字符串
        save_path: 保存路径，可选字符串
        **kwargs: 传递给降维算法的其他参数
    """
    set_paper_style()

    # 1. 降维处理
    n_samples, n_features = X.shape
    if n_features > 2:
        # print(f"[Analysis] Running {method.upper()} reduction from {n_features}d to 2d...")
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=2024, init='pca', learning_rate='auto', **kwargs)
        else:
            reducer = PCA(n_components=2, **kwargs)
        X_2d = reducer.fit_transform(X)
    else:
        X_2d = X

    # 2. 绘图
    plt.figure(figsize=(8, 6))

    # 使用 seaborn 处理颜色和图例，palette='tab10' 适合区分明显的类别
    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=labels,
        palette='tab10',
        s=60,  # 点的大小
        alpha=0.8,  # 透明度
        edgecolor='w',  # 点的白边，增加对比度
        legend='full'
    )

    # 3. 装饰
    plt.title(title if title else f'{method.upper()} Visualization (N={n_samples})')

    plt.xlabel(xlabel if xlabel else "")
    plt.ylabel(ylabel if ylabel else "")

    # 优化图例位置
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cluster')

    save_fig(plt.gcf(), save_path)
    # plt.show()


def plot_coassociation_heatmap(
        BPs: np.ndarray,
        Y: np.ndarray,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
):
    """
    绘制排序后的共协矩阵热力图
    """
    set_paper_style()

    # 1. 计算共协矩阵 (S = 1 - Hamming Distance)
    # BPs: (N, M)
    # print("[Analysis] Calculating Co-association Matrix...")
    D_hamming = pairwise_distances(BPs, metric='hamming')
    S = 1.0 - D_hamming  # 相似度矩阵 (0~1)

    # 2. 关键：根据真实标签 Y 对矩阵进行排序
    # 这样同类的样本就会聚在一起，形成对角线上的方块
    sort_indices = np.argsort(Y)
    S_sorted = S[sort_indices][:, sort_indices]

    # 3. 绘图
    plt.figure(figsize=(7, 6))

    # 使用 heatmap，颜色越深代表越相似
    sns.heatmap(
        S_sorted,
        cmap='viridis',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Co-association Probability'},
        rasterized=True
    )

    plt.title(title if title else f'Sorted Co-association Matrix (N={len(Y)})')

    plt.xlabel(xlabel if xlabel else "")
    plt.ylabel(ylabel if ylabel else "")

    save_fig(plt.gcf(), save_path)
    # plt.show()


def plot_metric_line(
        results_list: List[Dict[str, float]],
        metrics: Union[List[str], str] = 'ACC',
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
) -> None:
    """
    绘制多轮实验的折线图 (Trace Plot)。
    X轴为实验轮次(Run ID)，Y轴为指标得分。

    Args:
        results_list: 实验结果列表
        metrics: 要展示的指标，支持单个字符串或列表
    """
    set_paper_style()

    # 1. 参数标准化
    if isinstance(metrics, str):
        metrics = [metrics]

    # 2. 数据转换
    df = pd.DataFrame(results_list)
    n_runs = len(df)  # <--- 获取动态的实验轮数

    # 检查指标
    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        raise ValueError(f"None of {metrics} found in results.")

    # 3. 增加“轮次”列 (Run ID)
    df['Run ID'] = range(1, n_runs + 1)

    # 4. 转换长格式
    df_melt = df.melt(id_vars=['Run ID'], value_vars=valid_metrics,
                      var_name='Metric', value_name='Score')

    # 5. 绘图
    plt.figure(figsize=(8, 6))  # <--- 高度稍微调大一点(从5改为6)，给底部的文字留出空间

    sns.lineplot(
        data=df_melt,
        x='Run ID',
        y='Score',
        hue='Metric',
        style='Metric',
        markers=True,
        dashes=False,
        palette='tab10',
        linewidth=2,
        markersize=8
    )

    # 6. 装饰
    plt.title(title if title else f'Performance Trace over {n_runs} Runs')

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlabel(xlabel if xlabel is not None else "")
    plt.ylabel(ylabel if ylabel is not None else "")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 图例放外面
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # # =================================================================
    # # 新增：添加动态脚注说明
    # # =================================================================
    # caption_text = (
    #     f"Note: Results over {n_runs} independent runs, where each run uses a distinct non-overlapping subset of base partitions."
    # )
    #
    # plt.text(0.5, -0.10, caption_text,
    #          ha='center', va='top',
    #          transform=plt.gca().transAxes,
    #          fontsize=10, style='italic', color='black')
    # # =================================================================

    save_fig(plt.gcf(), save_path)
    # plt.show()


def plot_grid_heatmap(
        csv_path: str,
        x_param: str,
        y_param: str,
        metric: str = 'NMI',
        save_path: Optional[str] = None
) -> None:
    """
    绘制网格搜索热力图

    Args:
        csv_path: 网格搜索结果 CSV 文件的路径
        x_param: 用作 X 轴的参数列名（如 'nBase'）
        y_param: 用作 Y 轴的参数列名（如 'k'）
        metric: 用作热力图颜色的指标列名，默认 'NMI'
        save_path: 保存路径，可选字符串
    """
    set_paper_style()

    # 1. 读取数据
    df = pd.read_csv(csv_path)

    # 2. 数据透视表 (Pivot)
    # 计算每个 (x, y) 组合的 metric 均值（防止有重复实验）
    pivot_table = df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc='mean')

    # 3. 绘图
    plt.figure(figsize=(8, 6))

    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis",
                cbar_kws={'label': metric})

    plt.title(f'Grid Search: {metric} vs ({x_param}, {y_param})')
    plt.xlabel(x_param)
    plt.ylabel(y_param)

    # 翻转Y轴，让坐标原点在左下角（符合通常直觉）
    plt.gca().invert_yaxis()

    save_fig(plt.gcf(), save_path)
    plt.show()

