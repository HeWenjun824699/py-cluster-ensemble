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
        show: Optional[bool] = True,
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

    if show:
        plt.show()
    else:
        plt.close()


def plot_coassociation_heatmap(
        BPs: np.ndarray,
        Y: np.ndarray,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: Optional[bool] = True
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

    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_line(
        results_list: List[Dict[str, float]],
        metrics: Union[List[str], str] = 'ACC',
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: Optional[bool] = True
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

    if show:
        plt.show()
    else:
        plt.close()


def plot_parameter_sensitivity(
        csv_file: str,
        target_param: str,
        metric: Optional[str] = 'NMI',
        fixed_params: Optional[Dict[str, Any]] = None,
        method_name: Optional[str] = None,  # 如果csv含多种方法，需指定一种
        save_path: Optional[str] = None,
        show: Optional[bool] = True
):
    """
    绘制单参数敏感性折线图（控制变量法）。

    Args:
        target_param: 研究的参数 (例如 't')
        metric: Y轴: 评价指标
        fixed_params: 用户手动指定的固定参数
        method_name: 如果csv含多种方法，需指定一种

    逻辑：
    1. 锁定特定算法。
    2. 确定除了 target_param 以外还有哪些参数在变化。
    3. 确定这些背景参数的固定值：
       - 如果用户在 fixed_params 里指定了，就用用户的。
       - 如果没指定，就自动选择该方法在全局最优结果下的参数值 (Best Practice)。
    4. 筛选数据并绘图。
    """

    # 1. 读取数据
    df = pd.read_csv(csv_file)

    # 2. 筛选特定算法
    # if method_name:
    #     df = df[df['consensus_method'] == method_name]

    if df.empty:
        print(f"Error: No data found for method '{method_name}'")
        return

    # 3. 识别所有的超参数列 (你需要根据实际情况维护这个列表，或者自动检测)
    # 自动检测逻辑：列名不在黑名单中，且nunique > 1
    exclude_cols = {'Dataset', 'Exp_id', 'Status', 'Total_Time', 'consensus_method',
                    'ACC', 'NMI', 'ARI', 'Purity', 'AR', 'RI', 'MI', 'HI', 'F-Score',
                    'Precision', 'Recall', 'Entropy', 'SDCS', 'RME', 'Bal', 'Time'}

    potential_params = [c for c in df.columns if c not in exclude_cols]

    # 背景参数 = 所有潜在参数 - 目标参数
    background_params = [p for p in potential_params if p != target_param]

    # [建议新增] 检查用户是否传入了无效参数并给予提示
    if fixed_params:
        # 计算 CSV 里真正能用的参数集合
        valid_keys = set(background_params)
        user_keys = set(fixed_params.keys())

        # 找出用户传了但 CSV 里没有的参数
        ignored_keys = user_keys - valid_keys

        if ignored_keys:
            print(f"Warning: 以下固定参数在 CSV 中未找到或无法使用，已被忽略: {ignored_keys}")

    # 4. 确定固定值 (Context Context)
    current_fixed = {}

    # 先找到全局最优的那一行 (作为默认基准)
    # idxmax 返回最大值的索引
    best_row_idx = df[metric].idxmax()
    best_row = df.loc[best_row_idx]

    for param in background_params:
        # A. 用户指定了 -> 用用户的
        if fixed_params and param in fixed_params:
            val = fixed_params[param]
            current_fixed[param] = val
        # B. 用户没指定 -> 用最优行的数据 (Auto-Best)
        else:
            # 注意：如果某参数列全是NaN (如 mcla 的 t)，直接忽略
            if pd.isna(best_row[param]):
                continue
            current_fixed[param] = best_row[param]

    # 5. 构建筛选条件 Query
    query_parts = []
    for param, val in current_fixed.items():
        # 处理字符串还是数值的查询差异
        if isinstance(val, str):
            query_parts.append(f"{param} == '{val}'")
        else:
            query_parts.append(f"{param} == {val}")

    query_str = " & ".join(query_parts)

    # 6. 执行筛选
    if query_str:
        plot_df = df.query(query_str).copy()
    else:
        plot_df = df.copy()  # 没有背景参数，直接画

    # 排序，保证折线连贯
    if not plot_df.empty:
        plot_df = plot_df.sort_values(by=target_param)
    else:
        print(f"Error: 找不到符合条件的数据组合: {current_fixed}")
        print("建议检查 fixed_params 是否在网格搜索空间内。")
        return

    # 7. 绘图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x=target_param, y=metric, marker='o', linewidth=2)

    # 标题生成：展示我们固定了什么
    fixed_info = ", ".join([f"{k}={v}" for k, v in current_fixed.items()])
    if fixed_info:
        plt.title(f"Sensitivity of {target_param} on {metric}\n(Fixed: {fixed_info})")
    else:
        plt.title(f"Sensitivity of {target_param} on {metric}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylabel(metric)
    plt.xlabel(target_param)

    save_fig(plt.gcf(), save_path)

    if show:
        plt.show()
    else:
        plt.close()
