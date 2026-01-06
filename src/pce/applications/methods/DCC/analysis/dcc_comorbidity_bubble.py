import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

def get_num_proportion(data):
    """
    Extracts numerical values and proportions from string data.

    Parameters
    ----------
    data : array-like
        Input data containing strings in the format "number(proportion)".

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        - The extracted numerical values (n).
        - The extracted proportions (p).
    """
    data = np.array(data)
    n, p = [], []
    for line in data:
        n_term, p_term = [], []
        for item in line:
            # item is expected to be string like "123(45.6)"
            item = str(item)
            if '(' in item and item.endswith(')'):
                idx = item.find('(')
                try:
                    val_n = float(item[:idx])
                    val_p = float(item[idx+1:-1])
                    n_term.append(val_n)
                    p_term.append(val_p)
                except ValueError:
                    n_term.append(0)
                    p_term.append(0)
            else:
                n_term.append(0)
                p_term.append(0)
        n.append(n_term)
        p.append(p_term)
    return np.array(n), np.array(p)

def dcc_comorbidity_bubble(csv_path, output_dir, output_filename='comorbidity_bubble.png'):
    """
    Generates a bubble chart from comorbidity data.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (expected format: first col labels, rest "n(p%)").
    output_dir : str
        Directory to save the plot.
    output_filename : str, optional
        Name of the output file. Default is 'comorbidity_bubble.png'.

    Returns
    -------
    None
        The function saves the plot to the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        com = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    # Assuming first column is label, rest are data
    try:
        num, proportion = get_num_proportion(com.iloc[:, 1:].values)
    except Exception as e:
        print(f"Error parsing data: {e}")
        return

    colors = ['#3951a2', "#5c90c2", "#92c5de", "#fdb96b", "#f67948", "#da382a", "#a80326",
              '#66c2a5', '#fc8d62', '#8da0cb'] # extended palette
    
    fig, ax = plt.subplots(figsize=(10, 8)) # Added figsize
    
    # Plot bubbles
    for i in range(proportion.shape[0]):
        for j in range(proportion.shape[1]):
            if j < len(colors):
                color = colors[j]
            else:
                color = 'gray'
            
            # s is size, scaling by some factor, original was / 5
            ax.scatter(proportion[i, j], i, c=color, s=num[i][j] / 5, alpha=0.5)

    # Legend for clusters (S1, S2...)
    # Assuming columns correspond to S1, S2...
    n_clusters = proportion.shape[1]
    for i in range(min(n_clusters, len(colors))):
        ax.scatter([], [], c=colors[i], s=15, label=f'S{i + 1}')
    
    plt.legend(ncol=4, labelspacing=0.05, columnspacing=0.1, loc='upper center', bbox_to_anchor=(0.5, 1.1))

    # Legend for bubble sizes
    a = []
    sizes = [10, 100, 1000]
    for area in sizes:
        a.append(ax.scatter([], [], c='k', alpha=0.3, s=area / 5, label=str(area)))
    
    leg = Legend(ax, a, [str(s) for s in sizes],
                 loc='lower right', ncol=3, labelspacing=0.05, columnspacing=0.5)
    ax.add_artist(leg)

    plt.xlabel('Proportion (%)')
    plt.yticks([i for i in range(com.shape[0])], com.iloc[:, 0])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, right=False)
    
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Comorbidity bubble chart saved to {save_path}")
