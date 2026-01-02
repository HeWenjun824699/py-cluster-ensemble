import json
import pce


# 场景 A: 绘制原始数据的 T-SNE 散点图
X, Y = pce.io.load_mat_X_Y('./data/10X_PBMC_4271n_16653d_8c.mat')
pce.analysis.plot_2d_scatter(
    X, Y,
    xlabel='Dimension 1',
    ylabel='Dimension 2',
    method='tsne',
    title='10X_PBMC_4271n_16653d_8c Ground Truth',
    save_path='./data/png/X_Y_tsne.png'
)

# 场景 B: 绘制共协矩阵热力图
BPs, Y = pce.io.load_mat_BPs_Y('./data/generators/CDKM200/10X_PBMC_4271n_16653d_8c_CDKM200.mat')
pce.analysis.plot_coassociation_heatmap(
    BPs, Y,
    xlabel='Sample Index (Sorted by Ground Truth)',
    ylabel='Sample Index (Sorted by Ground Truth)',
    title='10X_PBMC_4271n_16653d_8c_CDKM200 HeatMap',
    save_path='./data/png/BPs_Y_heatmap.png'
)

# 场景 C: 绘制聚类指标折线图
with open('./data/grid/GridSearch_001/10X_PBMC_4271n_16653d_8c_CDKM200/Exp_001_LWEA/metrics.json') as f:
    results_list = json.loads(f.read())
# metrics = ["ACC", "NMI", "Purity", "AR", "RI", "MI", "HI", "F-Score", "Precision", "Recall", "Entropy", "SDCS", "RME", "Bal"]
metrics = ["ACC", "NMI", "Purity", "AR", "RI", "MI", "HI", "F-Score", "Precision", "Recall", "Entropy", "RME", "Bal"]
pce.analysis.plot_metric_line(
    results_list=results_list,
    metrics=metrics,
    xlabel='Experiment Run ID',
    ylabel='Score',
    title='10X_PBMC_4271n_16653d_8c_CDKM200 Metrics',
    save_path='./data/png/line_plot.png'
)

# 场景 D: 绘制参数敏感度分析折线图
csv_file = './data/grid/GridSearch_001/10X_PBMC_4271n_16653d_8c_CDKM200/10X_PBMC_4271n_16653d_8c_CDKM200.csv'
pce.analysis.plot_parameter_sensitivity(
    csv_file,
    target_param='theta',
    metric='ACC',
    fixed_params={},
    method_name='lwea',
    save_path='./data/png/parameter_sensitivity.png',
    show=True,
    show_values=True
)
