import os
import pickle
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import correspond, worker_kmeans_dynamic
from ....consensus.dcc import dcc


def run_consensus_clustering(input_path, output_path, hidden_dims, k_min=3, k_max=10, **kwargs):
    """
    Args:
        input_path: Dataset root directory
        output_path: Output root directory (including representations and results)
        hidden_dims: list, list of hidden_dims used during representation generation
        k_min, k_max: Range of cluster numbers
    """

    cfg = {
        'seed': 2026
    }
    cfg.update(kwargs)
    args = SimpleNamespace(**cfg)

    # 1. Path preparation
    rep_folder = os.path.join(output_path, "representations")
    res_folder = os.path.join(output_path, "results")
    pkl_folder = os.path.join(res_folder, "pkls")
    png_folder = os.path.join(res_folder, "pngs")
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    if not os.path.exists(pkl_folder):
        os.makedirs(pkl_folder)
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # 2. Load Ground Truth (for sorting)
    data_file = os.path.join(input_path, 'data.pkl')
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    y = np.array(data[1])

    # 3. Initialize variables
    cdf = []
    areas = []
    consensus_bars = []

    # Ensure hidden_dims is a list (convert to list if it's a range object)
    hidden_dims_list = list(hidden_dims)
    n_estimators = len(hidden_dims_list)

    print(f"\nStarting Consensus Clustering k={k_min}-{k_max}, models={n_estimators}...")

    plt.figure()

    # 4. Main loop: iterate through different K values
    for k in range(k_min, k_max):
        print(f"  Processing k={k}...")

        # --- A. Generate Base Clustering ---
        # Here i corresponds to each hidden_dim in hidden_dims_list
        clusters = [worker_kmeans_dynamic(rep_folder, k, i, seed=args.seed) for i in
                    tqdm(hidden_dims_list, desc=f"Base Clustering k={k}")]

        """
        Old Logic:
        
        clusters = np.array(clusters)  # Shape: [num_models, num_samples]

        # --- B. Calculate Consensus Matrix ---
        # Changed to single-threaded execution: avoid serialization overhead and deadlock risks when transferring large arrays in multiprocessing
        print(f"  Calculating Consensus Matrix for k={k}...")

        # Use List Comprehension directly with tqdm to show progress
        m = [worker_m(clusters, i) for i in tqdm(range(clusters.shape[1]), desc="Building Matrix")]

        # Normalize: divide by the number of base clustering models (i.e., count of hidden_dims)
        m = np.array(m, dtype=np.float32) / len(hidden_dims_list)

        # Save original consensus matrix
        with open(os.path.join(res_folder, 'pkls', f'm_{k}.pkl'), 'wb') as f:
            pickle.dump(m, f)

        # --- C. Final Clustering on Matrix ---
        model = KMeans(n_clusters=k, random_state=args.seed)
        model.fit(m)
        res = np.array(model.labels_)

        # Sort by mortality rate (re-labeling)
        res = correspond(res, y)
        """

        # Convert to format required by dcc_consensus (n_samples, n_estimators)
        BPs = np.array(clusters).T

        # --- B & C. Replaced with dcc_consensus call ---
        # We set nRepeat=1 because the logic here is for a deterministic analysis of a specific k
        print(f"  Running DCC Consensus for k={k}...")
        labels_list, _, m = dcc(
            BPs=BPs,
            Y=None,
            nClusters=k,  # Current loop k
            nBase=n_estimators,  # Number of base clusters
            nRepeat=1,
            seed=args.seed,
            return_matrix=True
        )

        # Get result (take the first one since nRepeat=1)
        res = labels_list[0]

        # Save original consensus matrix (keep original logic)
        with open(os.path.join(res_folder, 'pkls', f'm_{k}.pkl'), 'wb') as f:
            pickle.dump(m, f)

        # --- Post-processing: Sort by Ground Truth (Re-labeling) ---
        res = correspond(res, y)

        s = sorted(res)
        with open(os.path.join(res_folder, 'pkls', f'consensus_cluster_{k}.pkl'), 'wb') as f:
            pickle.dump(res, f)

        # --- D. Calculate statistical metrics (CDF, PAC, etc.) ---
        # Calculate cluster boundaries (for visualization)
        c_num = []
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                c_num.append(i)
        c_num.append(len(s))

        # Sort matrix for visualization (optional, mainly for metric calculation here)
        m = m[:, res.argsort()]
        m = m[res.argsort(), :]

        # Calculate Consensus Value (for bar plot)
        b = []
        for i in range(len(c_num)):
            if i == 0:
                area_sum = m[:c_num[0], :c_num[0]].sum()
                count = c_num[0] * c_num[0]
                b.append(area_sum / count if count > 0 else 0)
            else:
                area_sum = m[c_num[i - 1]:c_num[i], c_num[i - 1]:c_num[i]].sum()
                count = (c_num[i] - c_num[i - 1]) ** 2
                b.append(area_sum / count if count > 0 else 0)
        consensus_bars.append(b)

        # Calculate CDF
        consensus_value = m.ravel()
        hist, bin_edges = np.histogram(consensus_value, bins=100, range=(0, 1))
        # Correct histogram statistics (exclude diagonal or tiny errors, keeping original logic here)
        # hist[-1] -= m.shape[0]

        c = np.cumsum(hist / sum(hist))
        cdf.append(c)

        # Plot: CDF Curve
        width = (bin_edges[1] - bin_edges[0])
        plt.plot(bin_edges[1:] - width / 2, c, label=f'k={k}')

        # Calculate Area Under CDF
        # Use simple rectangle approximation or trapezoidal formula here
        delta_a = [h * width for h in c]  # Simplified calculation
        a = np.sum(delta_a)
        areas.append(a)

    # 5. Save charts
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel('Consensus Index')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Consensus Values')
    plt.savefig(os.path.join(res_folder, 'pngs', 'cdf.png'), dpi=300)
    plt.close()

    # Delta K Plot
    delta_k = []
    for i in range(len(areas)):
        if i == 0:
            delta_k.append(areas[0])  # Or 0, depending on definition
        else:
            # Relative change calculation
            if areas[i - 1] != 0:
                delta_k.append((areas[i] - areas[i - 1]) / areas[i - 1])
            else:
                delta_k.append(0)

    plt.figure()
    k_range = range(k_min, k_max)
    plt.plot([i for i in k_range], delta_k)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Relative Change')
    plt.title('Relative Change in Area Under CDF')
    plt.xticks([i for i in k_range])
    plt.savefig(os.path.join(res_folder, 'pngs', 'delta.png'), dpi=300)
    plt.close()

    # Average Consensus Plot
    plt.figure()
    index = 0
    i_s = []
    for i in range(len(consensus_bars)):
        k_val = k_range[i]
        b = consensus_bars[i]
        x = [j for j in range(index, index + len(b))]
        plt.bar(x, b, label=f'k={k_val}')
        i_s.append(index + len(b) * 0.5 - 0.5)
        index += len(b) + 2

    plt.xticks(ticks=i_s, labels=[f"k={i}" for i in k_range])
    plt.xlabel('Subphenotypes')
    plt.ylabel('Average Consensus Value')
    plt.title('Average Consensus Value')
    plt.savefig(os.path.join(res_folder, 'pngs', 'aver_con.png'), dpi=300)
    plt.close()

    print(f"Consensus Clustering Completed. Results saved to {res_folder}")
