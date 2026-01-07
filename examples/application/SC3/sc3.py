import os

from pce.applications.sc3 import sc3_application
from pce.io.load_rda import load_rda_X_Y
from pce.metrics.evaluation_single import evaluation_single
from pce.io.save_results import save_results_csv, save_results_xlsx, save_results_mat


# 1. Prepare Data
X, Y, gene_names, cell_names = load_rda_X_Y(f"data/yan.rda")
print(f"\nData loaded: {X.shape[0]} cells, {X.shape[1]} genes.")

# 2. Run SC3-Nature methods-2017
print("\nRunning SC3-Nature methods-2017...")
output_dir = "results"
labels, biology_res, time_cost = sc3_application(
    X=X,
    Y=None,
    nClusters=None,
    gene_names=gene_names,
    cell_names=cell_names,
    output_directory=output_dir,
    biology=True,
    gene_filter=True,
    seed=2026
)

# 3. Evaluate
res = evaluation_single(labels, Y, time_cost)
print(f"ACC: {res.get('ACC'):.4f}")
print(f"NMI: {res.get('NMI'):.4f}")
print(f"ARI: {res.get('AR'):.4f}")

# 4. Save Results
save_results_csv(res, output_dir, default_name="metrics.csv")
print(f"\nResults exported to: {os.path.abspath(output_dir)}")
