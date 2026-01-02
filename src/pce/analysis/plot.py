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
    Plot a 2D scatter plot for feature data or clustering results.

    If the input feature matrix has more than two dimensions, this function
    automatically applies dimensionality reduction (t-SNE or PCA) before plotting.
    It uses a paper-style aesthetic and a high-contrast color palette.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Label vector of shape (n_samples,). Used to determine the color of
        each scatter point.
    xlabel : str, optional
        Label for the X-axis. If None, no label is shown.
    ylabel : str, optional
        Label for the Y-axis. If None, no label is shown.
    method : str, default='tsne'
        Dimensionality reduction method. Supports 'tsne' or 'pca'.
    title : str, optional
        The title of the plot. If None, a default title including the method
        name and sample size is generated.
    save_path : str, optional
        Path to save the generated figure. Supports formats like '.png'
        (bitmap) and '.pdf' (vector, recommended for papers).
    show : bool, default=True
        Whether to display the plot window after generation.
    **kwargs : Any
        Additional arguments passed to the dimensionality reduction algorithm
        (e.g., perplexity for t-SNE).

    Returns
    -------
    None
    """

    set_paper_style()

    # 1. Dimensionality reduction
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

    # 2. Plotting
    plt.figure(figsize=(8, 6))

    # Use seaborn for colors and legends, palette='tab10' is suitable for distinct categories
    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=labels,
        palette='tab10',
        s=60,  # Dot size
        alpha=0.8,  # Transparency
        edgecolor='w',  # White edge for dots to increase contrast
        legend='full'
    )

    # 3. Decoration
    plt.title(title if title else f'{method.upper()} Visualization (N={n_samples})')

    plt.xlabel(xlabel if xlabel else "")
    plt.ylabel(ylabel if ylabel else "")

    # Optimize legend position
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
    Plot a sorted co-association matrix heatmap to visualize ensemble structure.

    The function calculates the co-association matrix from base partitions and
    reorders it based on the ground truth (or reference) labels. This reordering
    places similar samples in adjacent rows/columns, ideally forming distinct
    diagonal blocks if the ensemble is consistent.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators).
    Y : np.ndarray
        Ground truth or reference label vector of shape (n_samples,).
        Crucial for reordering the matrix to reveal cluster structures.
    xlabel : str, optional
        Label for the X-axis.
    ylabel : str, optional
        Label for the Y-axis.
    title : str, optional
        The title of the plot. Defaults to 'Sorted Co-association Matrix'.
    save_path : str, optional
        Path to save the plot.
    show : bool, default=True
        Whether to display the heatmap window.

    Returns
    -------
    None
    """

    set_paper_style()

    # 1. Calculate Co-association Matrix (S = 1 - Hamming Distance)
    # BPs: (N, M)
    # print("[Analysis] Calculating Co-association Matrix...")
    D_hamming = pairwise_distances(BPs, metric='hamming')
    S = 1.0 - D_hamming  # Similarity matrix (0~1)

    # 2. Key: Sort matrix based on true labels Y
    # So that samples of the same class cluster together, forming diagonal blocks
    sort_indices = np.argsort(Y)
    S_sorted = S[sort_indices][:, sort_indices]

    # 3. Plotting
    plt.figure(figsize=(7, 6))

    # Use heatmap, darker colors indicate higher similarity
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
    Plot a trace plot of performance metrics over multiple experimental runs.

    Useful for visualizing the stability and convergence of ensemble algorithms
    across independent repetitions (nRepeat). Each metric is represented by a
    distinct line with markers.

    Parameters
    ----------
    results_list : List[Dict[str, float]]
        A list of dictionaries containing evaluation metrics for each run,
        typically the output of `evaluation_batch`.
    metrics : Union[List[str], str], default='ACC'
        The specific metrics to plot (e.g., 'ACC', 'NMI', 'ARI'). Can be
        a single string or a list of multiple metrics.
    xlabel : str, optional
        Label for the X-axis (e.g., 'Run ID').
    ylabel : str, optional
        Label for the Y-axis (e.g., 'Score').
    title : str, optional
        The title of the plot.
    save_path : str, optional
        Path to save the resulting line chart.
    show : bool, default=True
        Whether to display the plot window.

    Returns
    -------
    None
    """

    set_paper_style()

    # 1. Parameter standardization
    if isinstance(metrics, str):
        metrics = [metrics]

    # 2. Data conversion
    df = pd.DataFrame(results_list)
    n_runs = len(df)  # <--- Get dynamic number of experimental runs

    # Check metrics
    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        raise ValueError(f"None of {metrics} found in results.")

    # 3. Add "Run ID" column
    df['Run ID'] = range(1, n_runs + 1)

    # 4. Convert to long format
    df_melt = df.melt(id_vars=['Run ID'], value_vars=valid_metrics,
                      var_name='Metric', value_name='Score')

    # 5. Plotting
    plt.figure(figsize=(8, 6))  # <--- Increase height slightly (from 5 to 6) to leave space for text at the bottom

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

    # 6. Decoration
    plt.title(title if title else f'Performance Trace over {n_runs} Runs')

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlabel(xlabel if xlabel is not None else "")
    plt.ylabel(ylabel if ylabel is not None else "")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # # =================================================================
    # # Added: Add dynamic footnote explanation
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
        method_name: Optional[str] = None,  # If CSV contains multiple methods, specify one
        save_path: Optional[str] = None,
        show: Optional[bool] = True,
        show_values: Optional[bool] = True
):
    """
    Plot a sensitivity analysis line chart for a specific hyperparameter.

    Based on grid search results, this function implements the 'Control Variable
    Method' by fixing background parameters and varying a target parameter to
    observe its effect on a chosen performance metric.

    Parameters
    ----------
    csv_file : str
        Path to the summary CSV file generated by `GridSearcher`.
    target_param : str
        The hyperparameter name to be analyzed on the X-axis (e.g., 't', 'k', 'alpha').
    metric : str, default='NMI'
        The evaluation metric to be plotted on the Y-axis.
    fixed_params : Dict[str, Any], optional
        A dictionary of specific background parameters to fix. If None, the
        function automatically selects the parameter values corresponding
        to the global best result.
    method_name : str, optional
        Filter results for a specific consensus algorithm if the CSV contains
        multiple methods.
    save_path : str, optional
        Path to save the sensitivity chart.
    show : bool, default=True
        Whether to display the plot window.
    show_values : bool, default=True
        If True, displays the specific metric values as text labels above
        each data point.

    Returns
    -------
    None
    """

    # 1. Read data
    df = pd.read_csv(csv_file)

    # 2. Filter specific algorithm
    # if method_name:
    #     df = df[df['consensus_method'] == method_name]

    if df.empty:
        print(f"Error: No data found for method '{method_name}'")
        return

    # 3. Identify all hyperparameter columns (you need to maintain this list or auto-detect)
    # Auto-detection logic: Column name not in blacklist and nunique > 1
    exclude_cols = {'Dataset', 'Exp_id', 'Status', 'Total_Time', 'consensus_method',
                    'ACC', 'NMI', 'ARI', 'Purity', 'AR', 'RI', 'MI', 'HI', 'F-Score',
                    'Precision', 'Recall', 'Entropy', 'SDCS', 'RME', 'Bal', 'Time'}

    potential_params = [c for c in df.columns if c not in exclude_cols]

    # Background parameters = All potential parameters - Target parameter
    background_params = [p for p in potential_params if p != target_param]

    # [Suggested Addition] Check if user passed invalid parameters and provide hint
    if fixed_params:
        # Calculate valid parameter set in CSV
        valid_keys = set(background_params)
        user_keys = set(fixed_params.keys())

        # Find parameters passed by user but not in CSV
        ignored_keys = user_keys - valid_keys

        if ignored_keys:
            print(f"Warning: The following fixed parameters were not found in CSV or cannot be used, and have been ignored: {ignored_keys}")

    # 4. Determine fixed values (Context Context)
    current_fixed = {}

    # Find the global best row first (as default baseline)
    # idxmax returns index of maximum value
    best_row_idx = df[metric].idxmax()
    best_row = df.loc[best_row_idx]

    for param in background_params:
        # A. User specified -> Use user's
        if fixed_params and param in fixed_params:
            val = fixed_params[param]
            current_fixed[param] = val
        # B. User not specified -> Use data from best row (Auto-Best)
        else:
            # Note: If a parameter column is all NaN (e.g., t in mcla), ignore it
            if pd.isna(best_row[param]):
                continue
            current_fixed[param] = best_row[param]

    # 5. Build filter condition Query
    query_parts = []
    for param, val in current_fixed.items():
        # Handle query difference between string and number
        if isinstance(val, str):
            query_parts.append(f"{param} == '{val}'")
        else:
            query_parts.append(f"{param} == {val}")

    query_str = " & ".join(query_parts)

    # 6. Execute filter
    if query_str:
        plot_df = df.query(query_str).copy()
    else:
        plot_df = df.copy()  # No background parameters, plot directly

    # Sort to ensure continuous line
    if not plot_df.empty:
        plot_df = plot_df.sort_values(by=target_param)
    else:
        print(f"Error: No data combination found matching condition: {current_fixed}")
        print("Suggest checking if fixed_params is within grid search space.")
        return

    # 7. Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x=target_param, y=metric, marker='o', linewidth=2)

    # =========== [Modification Start: Add value labels] ===========
    if show_values:
        for index, row in plot_df.iterrows():
            plt.text(
                x=row[target_param],
                y=row[metric],
                s=f"{row[metric]:.4f}",  # Format string: Keep 4 decimal places
                ha='center',  # Horizontal alignment: Center
                va='bottom',  # Vertical alignment: Above the point
                fontsize=10,  # Font size
                color='black'
            )
    # =========== [Modification End] ========================

    # Title generation: Show what we fixed
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
