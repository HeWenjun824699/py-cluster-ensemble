import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dcc_relative_risk(csv_path, output_dir, output_filename='relative_risk.png'):
    """
    Generates a forest plot for Relative Risk.
    
    Args:
        csv_path: Path to the CSV file.
        output_dir: Directory to save the plot.
        output_filename: Name of the output file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    # Check for RR, Low, High
    # Case insensitive check
    cols = {c.lower(): c for c in data.columns}
    if 'rr' in cols and 'low' in cols and 'high' in cols:
        rr = data[cols['rr']]
        low = data[cols['low']]
        high = data[cols['high']]
    else:
        # Fallback: assume columns 4, 5, 6 (0-indexed 3, 4, 5) if available, 
        # as R used data[,1:3] for text, so next might be numbers.
        # But R allows naming arguments `RR=...`.
        # Let's try to find numeric columns if names don't match.
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            rr = data[numeric_cols[0]]
            low = data[numeric_cols[1]]
            high = data[numeric_cols[2]]
        else:
            print("Error: Could not identify RR, Low, High columns.")
            return

    # Labels: R uses columns 1:3. I'll concatenate them or use the first one.
    # Let's use the first column as the primary label.
    labels = data.iloc[:, 0].astype(str).values
    
    # Filter out NaNs and Infs
    mask = np.isfinite(rr) & np.isfinite(low) & np.isfinite(high)
    if not mask.all():
        print(f"Warning: Dropping {np.sum(~mask)} rows with non-finite values.")
        rr = rr[mask]
        low = low[mask]
        high = high[mask]
        labels = labels[mask]
        
    if len(rr) == 0:
        print("Error: No valid data points to plot.")
        return
    
    n_items = len(labels)
    y_pos = np.arange(n_items)

    fig, ax = plt.subplots(figsize=(10, max(6, n_items * 0.5)))

    # Plot Error Bars
    # xerr expects relative error, so subtract from mean
    xerr = [rr - low, high - rr]
    
    # Colors matching R styles if possible, else distinct
    # R has a list of colors. Let's use a standard cycle or specific map if implied.
    # R code had specific colors for rows?
    # styles <- fpShapesGp(...) with list of colors.
    # It seems to map to rows.
    # I'll just use black/blue for simplicity or iterate colors.
    
    ax.errorbar(rr, y_pos, xerr=xerr, fmt='s', color='black', ecolor='gray', capsize=3, markersize=5)
    
    # Add vertical line at 1 (Null hypothesis)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis() # Top to bottom
    
    ax.set_xlabel('Relative Risk (95% CI)')
    
    # Add text for RR (95% CI) on the right
    # Create a string "RR (Low-High)"
    # Calculate global max for text positioning
    max_val = np.max(high)
    for i, (r, l, h) in enumerate(zip(rr, low, high)):
        text = f"{r:.2f} ({l:.2f}-{h:.2f})"
        ax.text(max_val * 1.05, i, text, va='center', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Forest plot saved to {save_path}")
