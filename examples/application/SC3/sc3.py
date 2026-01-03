import os

from pce.applications.sc3 import sc3
from pce.io.load_rda import load_rda_X_Y
from pce.metrics.evaluation_single import evaluation_single
from pce.io.save_results import save_results_csv, save_results_xlsx, save_results_mat


def run_demo():
    # 1. Prepare Data
    # Path to the yan.rda file from the original R package structure
    rda_path = f"data/yan.rda"
    X, Y, gene_names, cell_names = load_rda_X_Y(rda_path)
    print(f"\nData loaded: {X.shape[0]} cells, {X.shape[1]} genes.")

    # 2. Run SC3-Nature methods-2017
    print("\nRunning SC3-Nature methods-2017...")
    output_dir = "results"

    labels, biology_res, time_cost = sc3(
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

    print(f"\nSC3-Nature methods-2017 finished in {time_cost:.2f} seconds.")

    # 3. Evaluate
    res = evaluation_single(labels, Y, time_cost)
    print(f"ACC (vs Ground Truth): {res.get('ACC'):.4f}")
    print(f"NMI (vs Ground Truth): {res.get('NMI'):.4f}")
    print(f"ARI (vs Ground Truth): {res.get('AR'):.4f}")

    # 4. Save Results
    save_results_csv(res, output_dir, default_name="metrics.csv")

    print("\nBiology Analysis Results:")
    if biology_res:
        if biology_res.get('de') is not None:
            print(f"- DE Genes calculated: {len(biology_res['de'])}")
        if biology_res.get('marker') is not None:
            print(f"- Marker Genes calculated: {len(biology_res['marker']['auroc'])}")
        if biology_res.get('outl') is not None:
            print(f"- Outlier Scores calculated: {len(biology_res['outl'])}")
    else:
        print("No biology results returned.")

    print(f"\nResults exported to: {os.path.abspath(output_dir)}")
    print("Check the Excel file in that directory.")


if __name__ == "__main__":
    run_demo()
