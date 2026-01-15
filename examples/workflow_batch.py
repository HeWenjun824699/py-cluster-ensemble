import pce


data_dir = r"./data/CDKM200"
output_dir = r"./results/workflow/batch"

# Multiple datasets with a single algorithm
pce.pipelines.consensus_batch(
    input_dir=data_dir,
    output_dir=output_dir,
    save_format="csv",
    consensus_method="cspa",
    seed=2026,
    nRepeat=10,
    overwrite=True
)

# # Multiple datasets and multiple algorithms
# consensus_methods = ["cspa", "ptaal", "lwea"]
# for method in consensus_methods:
#     pce.pipelines.consensus_batch(
#         input_dir=data_dir,
#         output_dir=output_dir,
#         save_format="csv",
#         consensus_method=method,
#         seed=2026,
#         nRepeat=10,
#         overwrite=True
#     )
