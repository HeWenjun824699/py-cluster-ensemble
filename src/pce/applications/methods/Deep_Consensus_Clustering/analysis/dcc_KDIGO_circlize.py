import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dcc_KDIGO_circlize(csv_path, output_dir, output_filename='KDIGO_circlize.png'):
    """
    Generates a Heatmap (proxy for Chord Diagram) showing relationship between Stages and Clusters.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    output_dir : str
        Directory to save the plot.
    output_filename : str, optional
        Name of the output file. Default is 'KDIGO_circlize.png'.

    Returns
    -------
    None
        The function saves the plot to the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Try reading with header
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    # Check if first column is index (text) or data
    # R code set rownames to Stage1..3, so maybe the CSV doesn't have them or has them as col 1.
    # If first col is numeric, assume it's data.
    if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        # All numeric, assume purely data matrix
        data = df.values
        # If columns match S1..S7 length?
        # R sets colnames to S1..S7.
        col_labels = [f"S{i+1}" for i in range(data.shape[1])]
    else:
        # First col is likely index
        data = df.iloc[:, 1:].values
        col_labels = df.columns[1:]
        # Row labels from first col
        row_labels = df.iloc[:, 0].values

    # If row_labels not set above
    if 'row_labels' not in locals():
        row_labels = [f"Stage{i+1}" for i in range(data.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize? Or raw counts? Heatmap usually better with raw or row-normalized.
    # Let's show raw counts and color intensity.
    
    im = ax.imshow(data, cmap='Blues')
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Loop over data dimensions and create text annotations.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, int(data[i, j]),
                           ha="center", va="center", color="black" if data[i, j] < data.max()/2 else "white")

    ax.set_title("Overlap: KDIGO Stages vs Clusters")
    fig.tight_layout()
    
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {save_path}")
