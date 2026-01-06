import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dcc_KDIGO_dynamic(csv_path, output_dir, output_filename='KDIGO_dynamic.png'):
    """
    Generates a stacked bar chart (proxy for alluvial) showing AKI stage dynamics over time.
    
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

    # R code: aes(x = daydiff, stratum = data$aki_stage_max, y = freq)
    # The CSV structure is a bit weird in R: `data$data$aki_stage_max`? 
    # Let's assume standard CSV cols: daydiff, aki_stage_max, freq
    cols = {c.lower(): c for c in data.columns}
    
    # Map required columns
    day_col = cols.get('daydiff')
    stage_col = cols.get('aki_stage_max')
    freq_col = cols.get('freq')
    
    if not (day_col and stage_col and freq_col):
        # Fallback if names are different
        print(f"Error: Could not find required columns (daydiff, aki_stage_max, freq) in {data.columns}")
        return

    # Pivot: index=daydiff, columns=aki_stage_max, values=freq
    pivot_df = data.pivot_table(index=day_col, columns=stage_col, values=freq_col, aggfunc='sum').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors? Not specified in R snippet (uses fill=aki_stage_max default or implicitly palette).
    # Since stages are 1, 2, 3, maybe similar to Stage colors in previous plots?
    # Stage 1 (Grey), Stage 2 (Grey), Stage 3 (Grey)? Or distinct?
    # R plot used `fill=aki_stage_max`.
    # Let's pick distinct colors.
    colors_map = {
        '1': '#cccccc', '2': '#999999', '3': '#333333',
        1: '#cccccc', 2: '#999999', 3: '#333333',
        'Stage 1': '#cccccc', 'Stage 2': '#999999', 'Stage 3': '#333333'
    }
    # Or just use a colormap
    
    plot_colors = [colors_map.get(c, None) for c in pivot_df.columns]
    # If any None, replace with default cycle
    if any(c is None for c in plot_colors):
        plot_colors = None # Use default
    
    pivot_df.plot(kind='bar', stacked=True, ax=ax, color=plot_colors, width=0.8)

    plt.xlabel('Time after admission (hours)')
    # R labels: 0->24, 1->48, 2->72
    # Check index values
    if all(x in [0, 1, 2] for x in pivot_df.index):
        ax.set_xticklabels(['24', '48', '72'], rotation=0)
    
    plt.ylabel('Number of patients')
    plt.legend(title='AKI Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Dynamic AKI plot saved to {save_path}")
