from pce.applications import fast_ensemble

time_taken = fast_ensemble(
    input_file="data/sc_1.0_ring_cliques_100_10.tsv",
    output_file="results/results.csv",
    n_partitions=10,
    threshold=0.8,
    resolution=0.01,
    algorithm='leiden-cpm',
    relabel=False,
    delimiter=','
)

print(f"Time taken: {time_taken:.2f} seconds")
