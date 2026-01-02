import json
import pce


# Scenario A: Plot T-SNE scatter plot of raw data
X, Y = pce.io.load_mat_X_Y('./data/10X_PBMC_4271n_16653d_8c.mat')
pce.analysis.plot_2d_scatter(
    X, Y,
    xlabel='Dimension 1',
    ylabel='Dimension 2',
    method='tsne',
    title='10X_PBMC_4271n_16653d_8c Ground Truth',
    save_path='./results/png/X_Y_tsne.png'
)

# Scenario B: Plot co-association matrix heatmap
BPs, Y = pce.io.load_mat_BPs_Y('./data/CDKM200/10X_PBMC_4271n_16653d_8c_CDKM200.mat')
pce.analysis.plot_coassociation_heatmap(
    BPs, Y,
    xlabel='Sample Index (Sorted by Ground Truth)',
    ylabel='Sample Index (Sorted by Ground Truth)',
    title='10X_PBMC_4271n_16653d_8c_CDKM200 HeatMap',
    save_path='./results/png/BPs_Y_heatmap.png'
)

# Scenario C: Plot line chart of clustering metrics
with open('./results/grid/GridSearch_001/10X_PBMC_4271n_16653d_8c_CDKM200/Exp_001_LWEA/metrics.json') as f:
    results_list = json.loads(f.read())
# metrics = ["ACC", "NMI", "Purity", "AR", "RI", "MI", "HI", "F-Score", "Precision", "Recall", "Entropy", "SDCS", "RME", "Bal"]
metrics = ["ACC", "NMI", "Purity", "AR", "RI", "MI", "HI", "F-Score", "Precision", "Recall", "Entropy", "RME", "Bal"]
pce.analysis.plot_metric_line(
    results_list=results_list,
    metrics=metrics,
    xlabel='Experiment Run ID',
    ylabel='Score',
    title='10X_PBMC_4271n_16653d_8c_CDKM200 Metrics',
    save_path='./results/png/line_plot.png'
)

# Scenario D: Plot line chart for parameter sensitivity analysis
csv_file = './results/grid/GridSearch_001/10X_PBMC_4271n_16653d_8c_CDKM200/10X_PBMC_4271n_16653d_8c_CDKM200.csv'
pce.analysis.plot_parameter_sensitivity(
    csv_file,
    target_param='theta',
    metric='ACC',
    fixed_params={},
    method_name='lwea',
    save_path='./results/png/parameter_sensitivity.png',
    show=True,
    show_values=True
)
