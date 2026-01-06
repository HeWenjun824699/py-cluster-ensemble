import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_km(durations, events):
    """
    Calculates Kaplan-Meier survival curve and 95% Confidence Interval (Greenwood).

    Parameters
    ----------
    durations : array-like
        Array of time durations (time to event or censoring).
    events : array-like
        Array of event indicators (1 if event occurred, 0 if censored).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:
        - times: The time points.
        - survival: The survival probabilities.
        - ci_lower: The lower bound of the 95% confidence interval.
        - ci_upper: The upper bound of the 95% confidence interval.
    """
    df = pd.DataFrame({'T': durations, 'E': events}).sort_values('T')
    
    # Unique times where events happen
    unique_times = df['T'].unique()
    unique_times = np.sort(unique_times)
    
    n_at_risk = []
    n_events = []
    
    total = len(df)
    for t in unique_times:
        # At risk: T >= t
        at_risk = len(df[df['T'] >= t])
        # Events: T == t and E == 1
        event_count = len(df[(df['T'] == t) & (df['E'] == 1)])
        
        n_at_risk.append(at_risk)
        n_events.append(event_count)
        
    n_at_risk = np.array(n_at_risk)
    n_events = np.array(n_events)
    
    # Hazard = d / n
    # Survival = product(1 - d/n)
    # Handle division by zero
    hazard = np.divide(n_events, n_at_risk, out=np.zeros_like(n_events, dtype=float), where=n_at_risk!=0)
    survival = np.cumprod(1 - hazard)
    
    # Standard Error (Greenwood's formula)
    # sum( d / (n * (n-d)) )
    # Avoid div by zero in denominator
    denom = n_at_risk * (n_at_risk - n_events)
    var_term = np.divide(n_events, denom, out=np.zeros_like(n_events, dtype=float), where=denom>0)
    sum_var_term = np.cumsum(var_term)
    se = survival * np.sqrt(sum_var_term)
    
    ci_lower = np.maximum(0, survival - 1.96 * se)
    ci_upper = np.minimum(1, survival + 1.96 * se)
    
    # Prepend t=0, S=1
    times = np.concatenate(([0], unique_times))
    survival = np.concatenate(([1], survival))
    ci_lower = np.concatenate(([1], ci_lower))
    ci_upper = np.concatenate(([1], ci_upper))
    
    return times, survival, ci_lower, ci_upper

def dcc_survival_analysis(csv_path, output_dir, output_filename='survival_analysis.png'):
    """
    Generates Kaplan-Meier survival curves.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (must contain columns 'los', 'y', 'sub').
    output_dir : str
        Directory to save the plot.
    output_filename : str, optional
        Name of the output file. Default is 'survival_analysis.png'.

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
        
    required_cols = ['los', 'y', 'sub']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: CSV must contain {required_cols}")
        return

    groups = data['sub'].unique()
    
    # Sort groups: S1..S7, then Stage1..3
    # Helper to sort natural logic
    def sort_key(s):
        s = str(s)
        if s.startswith('S') and s[1:].isdigit():
            return (0, int(s[1:]))
        if s.startswith('Stage'):
            return (1, s)
        return (2, s)
        
    groups = sorted(groups, key=sort_key)
    
    # Colors: S1-S7 match earlier files, others grey
    colors_map = {
        'S1': '#3951a2', 'S2': '#5c90c2', 'S3': '#92c5de',
        'S4': '#fdb96b', 'S5': '#f67948', 'S6': '#da382a', 'S7': '#a80326',
        'Stage 1': '#cccccc', 'Stage 2': '#999999', 'Stage 3': '#333333',
        # Handle variations like "Stage1" vs "Stage 1"
        'Stage1': '#cccccc', 'Stage2': '#999999', 'Stage3': '#333333'
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for g in groups:
        sub_data = data[data['sub'] == g]
        T = sub_data['los'].values
        E = sub_data['y'].values
        
        times, surv, low, high = calculate_km(T, E)
        
        c = colors_map.get(str(g), 'black')
        
        # Step plot
        ax.step(times, surv, where='post', label=str(g), color=c, linewidth=1.5)
        # Shade CI
        ax.fill_between(times, low, high, step='post', color=c, alpha=0.1)
        
    plt.xlabel('Time (Day)')
    plt.ylabel('Survival Probability')
    plt.ylim(0, 1)
    plt.xlim(left=0)
    plt.legend(title='Subphenotypes', loc='lower left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Survival plot saved to {save_path}")
