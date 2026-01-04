from pce.applications import sc3, fast_ensemble


def test_sc3_basic(synthetic_data, tmp_path):
    X, Y = synthetic_data
    # Use a very small subset for speed
    X_small = X[:20, :5]  # 20 cells, 5 genes
    Y_small = Y[:20]
    
    output_dir = tmp_path / "sc3_out"
    
    # Run SC3
    labels, biology_res, time_cost = sc3(
        X_small, 
        Y=Y_small, 
        nClusters=3, 
        output_directory=str(output_dir),
        gene_filter=False,  # Disable filter for small data
        biology=False,      # Skip heavy biology steps
        n_cores=1,
        seed=2026,
        kmeans_nstart=10,  # Speed up
        kmeans_iter_max=100
    )
    
    assert labels.shape == (20,)
    assert isinstance(time_cost, float)
    # Check if output files were created (consensus matrix png etc)
    assert (output_dir / "png" / "consensus_matrix.png").exists()


def test_fast_ensemble(tmp_path):
    # Create a dummy edge list
    input_file = tmp_path / "network.txt"
    output_file = tmp_path / "communities.csv"
    
    # Create a clique of 0,1,2 and 3,4,5 connected by one edge
    edges = [
        "0 1", "1 2", "2 0",
        "3 4", "4 5", "5 3",
        "2 3"
    ]
    with open(input_file, 'w') as f:
        f.write("\n".join(edges))
        
    # Run FastEnsemble
    time_cost = fast_ensemble(
        input_file=str(input_file),
        output_file=str(output_file),
        n_partitions=5,
        algorithm='louvain'  # Faster than leiden usually or similar
    )
    
    assert output_file.exists()
    assert isinstance(time_cost, float)
    
    # Check output content
    import csv
    with open(output_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 6  # 6 nodes
