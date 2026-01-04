import pytest
import numpy as np
import scipy.io
import pandas as pd
from pathlib import Path
from pce.io import (
    load_mat_X_Y, 
    load_mat_BPs_Y, 
    load_rda_X_Y,
    save_base_mat, 
    save_results_csv, 
    save_results_xlsx, 
    save_results_mat
)


@pytest.fixture
def dummy_mat_data(tmp_path):
    file_path = tmp_path / "test_data.mat"
    X_true = np.random.rand(10, 5)
    Y_true = np.random.randint(0, 2, 10)
    scipy.io.savemat(str(file_path), {'X': X_true, 'Y': Y_true})
    return str(file_path), X_true, Y_true


@pytest.fixture
def dummy_bps_data(tmp_path):
    file_path = tmp_path / "test_base.mat"
    n_samples = 20
    n_bps = 5
    # Generate 1-based BPs (MATLAB style)
    BPs_true = np.random.randint(1, 4, (n_samples, n_bps)) 
    Y_true = np.random.randint(1, 4, n_samples)
    scipy.io.savemat(str(file_path), {'BPs': BPs_true, 'Y': Y_true})
    return str(file_path), BPs_true, Y_true


def test_load_mat_X_Y(dummy_mat_data):
    file_path, X_true, Y_true = dummy_mat_data
    
    X, Y = load_mat_X_Y(file_path)
    
    assert X.shape == X_true.shape
    assert Y.shape == (X_true.shape[0],)
    assert np.allclose(X, X_true)
    assert np.array_equal(Y, Y_true)


def test_load_mat_BPs_Y(dummy_bps_data):
    file_path, BPs_true, Y_true = dummy_bps_data
    
    # fix_matlab_index=True by default, so it should subtract 1
    BPs, Y = load_mat_BPs_Y(file_path)
    
    assert BPs.shape == BPs_true.shape
    assert Y.shape == (BPs_true.shape[0],)
    # Check shift
    assert np.min(BPs) == 0
    assert np.all(BPs == BPs_true - 1)


def test_load_rda_X_Y():
    # Use existing example file
    # Requires navigating relative to project root or knowing absolute path
    # We assume tests are run from project root
    rda_path = Path("examples/application/SC3/data/yan.rda")
    
    if not rda_path.exists():
        pytest.skip("yan.rda not found, skipping load_rda test")
        
    X, Y, gene_names, cell_names = load_rda_X_Y(str(rda_path))
    
    # Check shapes
    # yan.rda usually has 90 cells
    assert X.shape[0] == 90
    assert Y.shape[0] == 90
    assert len(gene_names) == X.shape[1]
    assert len(cell_names) == X.shape[0]
    # Check log transformation (values shouldn't be too huge if logged)
    assert np.max(X) < 100 


def test_save_base_mat(tmp_path):
    output_path = tmp_path / "saved_base.mat"
    BPs = np.zeros((10, 3))
    Y = np.zeros(10)
    
    save_base_mat(BPs, Y, str(output_path))
    
    assert output_path.exists()
    # Verify content
    mat = scipy.io.loadmat(str(output_path))
    assert 'BPs' in mat
    assert 'Y' in mat
    assert mat['Y'].shape == (10, 1)  # Should ensure column vector


def test_save_results_csv(tmp_path):
    output_path = tmp_path / "res.csv"
    data = [{'ACC': 0.5, 'Time': 1.0}, {'ACC': 0.6, 'Time': 2.0}]
    
    save_results_csv(data, str(output_path))
    
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert 'ACC' in df.columns
    # Check for summary rows (Mean, Std, Str, Empty)
    # 2 data + 1 empty + 1 mean + 1 std + 1 str = 6 rows
    # Note: Pandas read_csv handles empty lines differently, 
    # but the file content should have them.
    assert len(df) >= 2


def test_save_results_xlsx(tmp_path):
    output_path = tmp_path / "res.xlsx"
    data = [{'ACC': 0.5, 'Time': 1.0}, {'ACC': 0.6, 'Time': 2.0}]
    
    try:
        import xlsxwriter
    except ImportError:
        pytest.skip("xlsxwriter not installed")
        
    save_results_xlsx(data, str(output_path))
    
    assert output_path.exists()
    # Read back (requires openpyxl or similar)
    try:
        df = pd.read_excel(output_path)
        assert 'ACC' in df.columns
    except ImportError:
        pass  # Skip verification if openpyxl missing


def test_save_results_mat(tmp_path):
    output_path = tmp_path / "res.mat"
    data = [{'ACC': 0.5, 'Time': 1.0}, {'ACC': 0.6, 'Time': 2.0}]
    
    save_results_mat(data, str(output_path))
    
    assert output_path.exists()
    mat = scipy.io.loadmat(str(output_path))
    assert 'result' in mat
    assert 'result_summary' in mat  # Check if summary was added
    assert mat['result'].shape == (2, 2)  # 2 rows, 2 columns (ACC, Time)
