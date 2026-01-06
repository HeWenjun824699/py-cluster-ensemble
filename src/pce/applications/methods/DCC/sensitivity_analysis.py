import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from tqdm import tqdm

from .utils import correspond

def run_sensitivity_analysis(data_path, results_dir, k, output_dir, n_bootstrapping=1000, bootstrapping_proportion=0.7):
    """
    Performs sensitivity analysis using bootstrapping to evaluate cluster stability.
    
    Args:
        data_path: Path to the data pickle file (containing y labels).
        results_dir: Directory containing clustering results (cluster_{k}.pkl, m_{k}.pkl).
        k: Number of clusters.
        output_dir: Directory to save the sensitivity plot.
        n_bootstrapping: Number of bootstrap iterations.
        bootstrapping_proportion: Proportion of samples to resample.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Handle data_path being a directory
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, 'data.pkl')

    # Load data
    try:
        with open(data_path, 'rb') as f:
            # Assuming data.pkl contains [X, y] or similar structure where index 1 is labels
            # The original script did: y = pickle.load(...)[1]
            data_content = pickle.load(f)
            y = np.array(data_content[1])
            
        with open(os.path.join(results_dir, f'consensus_cluster_{k}.pkl'), 'rb') as f:
            sub = pickle.load(f)
            
        with open(os.path.join(results_dir, f'm_{k}.pkl'), 'rb') as f:
            rep = pickle.load(f)
            
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    ids = np.array(range(len(sub)))
    agg = np.array(rep)

    heat = np.zeros((k, k), dtype=np.float64)
    ari, nmi = [], []

    print(f"\nStarting Sensitivity Analysis (k={k}, n={n_bootstrapping})...")

    for i in tqdm(range(n_bootstrapping), desc='Bootstrapping'):
        boot = resample(ids, replace=False, n_samples=round(len(ids) * bootstrapping_proportion), random_state=i)
        s = sub[boot]
        yi = y[boot]

        # Subset consensus matrix for bootstrapped samples
        # The original script: m = agg[:, boot][boot, :] 
        # agg is (N, N) similarity matrix? Or representations?
        # In original script: rep = pickle.load(open(f'./results/m_{k}.pkl', 'rb'))
        # If m_{k} is the consensus matrix (NxN), then agg[:, boot][boot, :] makes sense (submatrix).
        m = agg[boot][:, boot] 

        model = KMeans(n_clusters=k, random_state=1) # n_jobs removed in newer sklearn
        model.fit(m)
        res = np.array(model.labels_)
        
        # Align labels
        res = correspond(res, yi)
        
        # Update heat map
        # Note: confusion_matrix might not be kxk if some clusters are missing in res or s
        # passing labels parameter ensures fixed shape
        cm = confusion_matrix(res, s, labels=range(k))
        heat += cm / len(s)

        ari.append(adjusted_rand_score(s, res))
        nmi.append(normalized_mutual_info_score(s, res))

    heat /= n_bootstrapping
    
    # Plotting
    plt.figure()
    sns.heatmap(heat, xticklabels=[i + 1 for i in range(k)], yticklabels=[i + 1 for i in range(k)])
    plt.xlabel('Subphenotypes of Patient Bootstrapping', fontsize=15)
    plt.ylabel('Derived Subphenotypes', fontsize=15)
    
    save_path = os.path.join(output_dir, f'sensitivity_k{k}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Sensitivity plot saved to {save_path}")

    # Statistics
    lo_ari, hi_ari = stats.norm.interval(0.95, loc=np.mean(ari), scale=np.std(ari))
    print(f'ARI: %.3f (95%% CI: %.3f-%.3f)' % (np.mean(ari), lo_ari, hi_ari))

    lo_nmi, hi_nmi = stats.norm.interval(0.95, loc=np.mean(nmi), scale=np.std(nmi))
    print(f'NMI: %.3f (95%% CI: %.3f-%.3f)' % (np.mean(nmi), lo_nmi, hi_nmi))
    
    return {'ari_mean': np.mean(ari), 'nmi_mean': np.mean(nmi)}
