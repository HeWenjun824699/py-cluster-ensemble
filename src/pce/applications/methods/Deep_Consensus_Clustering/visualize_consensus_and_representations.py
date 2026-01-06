import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_consensus_and_representations(results_dir, representations_dir, output_dir, k, hidden_dims):
    """
    Visualizes the consensus matrix and t-SNE of representations.
    
    Args:
        results_dir: Directory containing clustering results (consensus_cluster_{k}.pkl, m_{k}.pkl).
        representations_dir: Directory containing representation files (rep_{h}.pkl).
        output_dir: Directory to save the plots.
        k: Number of clusters (consensus k).
        hidden_dims: List of hidden dimensions to visualize.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Colors for plotting
    colors_palette = ['#3951a2', '#5c90c2', '#92c5de', '#fdb96b', '#f67948', '#da382a', '#a80326', 
                      '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'] 
    
    try:
        # Load consensus results
        with open(os.path.join(results_dir, f'consensus_cluster_{k}.pkl'), 'rb') as f:
            sub = pickle.load(f)
        sub = np.array(sub)
        
        with open(os.path.join(results_dir, f'm_{k}.pkl'), 'rb') as f:
            m = pickle.load(f)
            
    except Exception as e:
        print(f"Error loading consensus files: {e}")
        return

    # Calculate cluster boundaries
    s_sorted = sorted(sub)
    c_num = []
    for i in range(1, len(s_sorted)):
        if s_sorted[i] != s_sorted[i - 1]:
            c_num.append(i)
    c_num.append(len(s_sorted))

    # Sort consensus matrix
    sort_idx = sub.argsort()
    m = m[:, sort_idx]
    m = m[sort_idx, :]

    # Visualization of consensus matrix
    plt.figure()
    color = 'b'
    # Draw boundaries
    # Vertical/Horizontal lines for first block
    if len(c_num) > 0:
        plt.vlines(1, ymin=0, ymax=c_num[0], color=color, linewidths=3)
        plt.hlines(1, xmin=0, xmax=c_num[0], color=color, linewidths=3)
        
        # Last block
        if len(c_num) > 1:
            plt.vlines(len(s_sorted) - 1, ymin=c_num[-2], ymax=len(s_sorted) - 1, color=color, linewidths=3)
            plt.hlines(len(s_sorted) - 1, xmin=c_num[-2], xmax=len(s_sorted) - 1, color=color, linewidths=3)
            
        # Middle blocks
        for i in range(len(c_num) - 1):
            start = 0 if i == 0 else c_num[i-1] - 1
            end = c_num[i+1] - 1
            current = c_num[i] - 1
            
            # Simple boundary logic from original script
            # Adjusted slightly for safety
            if i == 0:
                 plt.vlines(c_num[i] - 1, 0, c_num[i + 1] - 1, color=color)
                 plt.hlines(c_num[i] - 1, 0, c_num[i + 1] - 1, color=color)
            else:
                plt.vlines(c_num[i] - 1, c_num[i - 1] - 1, c_num[i + 1] - 1, color=color)
                plt.hlines(c_num[i] - 1, c_num[i - 1] - 1, c_num[i + 1] - 1, color=color)

    sns.heatmap(m, cbar=False)  # Original didn't show cbar explicitly but default is yes, original had xticks/yticks []
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Patients')
    plt.ylabel('Patients')
    
    save_path_heap = os.path.join(output_dir, f'heap_{k}.png')
    plt.savefig(save_path_heap, dpi=300)
    plt.close()
    print(f"Consensus heatmap saved to {save_path_heap}")

    # Visualization of representations
    for h_dim in hidden_dims:
        rep_file = os.path.join(representations_dir, f'rep_{h_dim}.pkl')
        if not os.path.exists(rep_file):
            print(f"Representation file not found: {rep_file}, skipping.")
            continue
            
        try:
            with open(rep_file, 'rb') as f:
                r = pickle.load(f)
                
            tsne = TSNE(n_jobs=-1, n_components=2, init='pca', learning_rate='auto')
            r_tsne = tsne.fit_transform(r)
            
            plt.figure()
            plt.title(f'Hidden Dimensions={h_dim}')
            
            unique_clusters = np.unique(sub)
            for j_idx, j in enumerate(unique_clusters):
                r_tsne_s = r_tsne[sub == j]
                color_idx = j_idx % len(colors_palette)
                plt.scatter(r_tsne_s[:, 0], r_tsne_s[:, 1], label=f'S{j + 1}', c=colors_palette[color_idx], s=0.5)
            
            plt.legend()
            save_path_tsne = os.path.join(output_dir, f'tsne_h{h_dim}.png')
            plt.savefig(save_path_tsne, dpi=300)
            plt.close()
            print(f"t-SNE plot saved to {save_path_tsne}")
            
        except Exception as e:
            print(f"Error processing hidden dim {h_dim}: {e}")
