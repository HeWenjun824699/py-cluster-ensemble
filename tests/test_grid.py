import numpy as np
import scipy.io
import scipy.io

from pce.grid.grid_search import GridSearcher


def test_grid_search_basic(tmp_path):
    # 1. Setup Data
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    
    file_path = input_dir / "test_dataset.mat"
    X = np.random.rand(20, 5)
    Y = np.random.randint(0, 2, 20)
    scipy.io.savemat(str(file_path), {'X': X, 'Y': Y})
    
    # 2. Setup Grid Searcher
    gs = GridSearcher(str(input_dir), str(output_dir))
    
    # 3. Define small grid
    param_grid = {
        'consensus_method': 'cspa',
        'theta': [1, 5, 10, 20]
    }
    fixed_params = {
        'generator_method': 'litekmeans',
        'nPartitions': 5,
        'nRepeat': 1,
        'seed': 2026
    }
    
    # 4. Run
    gs.run(param_grid, fixed_params)
    
    # 5. Verify Output
    # Should create output_dir / test_dataset / Exp_001_CSPA / results.csv
    ds_dir = output_dir / "test_dataset"
    assert ds_dir.exists()
    
    exp_dir = next(ds_dir.glob("Exp_*_CSPA"))
    assert (exp_dir / "results.csv").exists()
    assert (exp_dir / "labels.csv").exists()
    assert (output_dir / "grid_summary.csv").exists()
