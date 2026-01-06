import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dcc_cluster_transfer(csv_path, output_dir, output_filename='cluster_transfer.png'):
    """
    Generates a stacked bar chart (proxy for alluvial) showing cluster evolution.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    output_dir : str
        Directory to save the plot.
    output_filename : str, optional
        Name of the output file. Default is 'cluster_transfer.png'.

    Returns
    -------
    None
        The function saves the plot to the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    # Expected columns based on R code: c, cluster, freq
    # R: aes(x = c, y = freq, fill = cluster)
    required_cols = ['c', 'cluster', 'freq']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: CSV must contain columns {required_cols}")
        # Try to infer if names are different or proceed if possible
        # For now, strict check or fallback
        return

    # Pivot data: index=c, columns=cluster, values=freq
    # Sum freq just in case there are multiple entries per c/cluster combo
    pivot_df = data.pivot_table(index='c', columns='cluster', values='freq', aggfunc='sum').fillna(0)

    # Colors from R code
    colors_map = {
        'S1': '#3951a2', 'S2': '#5c90c2', 'S3': '#92c5de',
        'S4': '#fdb96b', 'S5': '#f67948', 'S6': '#da382a', 'S7': '#a80326'
    }
    
    # Ensure columns are sorted or match S1..S7 if possible
    # filter cols that are in colors_map or assign default
    cols = sorted(pivot_df.columns)
    plot_colors = [colors_map.get(c, 'gray') for c in cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Stacked Bar
    pivot_df.plot(kind='bar', stacked=True, color=plot_colors, ax=ax, width=0.9)

    plt.xlabel('Number of clusters')
    plt.ylabel('Proportion of patients')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Cluster transfer plot saved to {save_path}")
