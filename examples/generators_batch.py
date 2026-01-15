import os
import pce


# ================= Configuration =================
DATA_DIR = r"./data"
OUT_ROOT = r"./data"
OVERWRITE = False

COMMON_ARGS = {
    "nClusters": None,
    "nPartitions": 200,
    "seed": 2026
}

METHODS = {
    "LKM": {
        "func": pce.generators.litekmeans,
        "args": {"maxiter": 100, "replicates": 1}
    },
    "CDKM": {
        "func": pce.generators.cdkmeans,
        "args": {"maxiter": 100, "replicates": 1}
    },
    "RSKM": {
        "func": pce.generators.rskmeans,
        "args": {"subspace_ratio": 0.5, "maxiter": 100, "replicates": 1}
    },
    "RPKM": {
        "func": pce.generators.rpkmeans,
        "args": {"projection_ratio": 0.5, "maxiter": 100, "replicates": 1}
    },
    "BAGKM": {
        "func": pce.generators.bagkmeans,
        "args": {"subsample_ratio": 0.8, "maxiter": 100, "replicates": 1}
    },
    "HETCLU": {
        "func": pce.generators.hetero_clustering,
        "args": {}
    },
    "SC3KM": {
        "func": pce.generators.sc3_kmeans,
        "args": {}
    },
    "SPECTRAL": {
        "func": pce.generators.spectral,
        "args": {}
    },
}

# ================= Main Logic =================
# 1. Iterate through .mat files
mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
print(f"Found {len(mat_files)} datasets: {mat_files}")

for filename in mat_files:
    print(f"\n>>> Processing Dataset: {filename}")

    # Load data
    data_path = os.path.join(DATA_DIR, filename)
    X, Y = pce.io.load_mat_X_Y(data_path)

    dataset_name = os.path.splitext(filename)[0]

    # 2. Iterate through generation methods
    for method_name, config in METHODS.items():
        folder_suffix = f"{method_name}{COMMON_ARGS['nPartitions']}"
        output_dir = os.path.join(OUT_ROOT, folder_suffix)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        out_filename = f"{dataset_name}_{folder_suffix}.mat"
        base_output_path = os.path.join(output_dir, out_filename)

        # Check for existing files
        if os.path.exists(base_output_path) and not OVERWRITE:
            print(f"    [Skip] {method_name} already exists: {out_filename}")
            continue

        print(f"    [Run] Generating {method_name}...")

        # Prepare arguments
        run_args = {
            "X": X,
            "Y": Y,
            **COMMON_ARGS,
            **config["args"]
        }

        # Execute
        BPs = config["func"](**run_args)

        # Save
        pce.io.save_base_mat(BPs, Y, base_output_path)
        print(f"          Saved to: {base_output_path}")

print("\nAll tasks completed.")
