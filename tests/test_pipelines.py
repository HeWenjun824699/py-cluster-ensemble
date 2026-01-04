import numpy as np
import scipy.io

from pce.pipelines import consensus_batch


def test_consensus_batch_end_to_end(tmp_path):
    # Setup directories
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    
    # Create dummy .mat file with X and Y
    file_path = input_dir / "test_data.mat"
    X = np.random.rand(20, 5)
    Y = np.random.randint(0, 2, 20)
    scipy.io.savemat(str(file_path), {'X': X, 'Y': Y})
    
    # Run consensus_batch
    # Use small parameters for speed
    consensus_batch(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        save_format="csv",
        consensus_method='cspa',
        generator_method='litekmeans',
        nPartitions=10,  # small number of partitions to generate
        seed=2026,
        nBase=2,    # small slice
        nRepeat=2   # small repeats
    )
    
    # Verify output
    expected_output_dir = output_dir / "CSPA"
    expected_output_file = expected_output_dir / "test_data_CSPA.csv"
    
    assert expected_output_dir.exists()
    assert expected_output_file.exists()
    
    # Check content briefly
    with open(expected_output_file, 'r') as f:
        content = f.read()
        assert "ACC" in content
