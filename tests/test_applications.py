import numpy as np
import pytest
from unittest.mock import patch
from pce.applications import sc3
from pce.applications.dcc import dcc_application
from pce.applications.icsc import icsc_mul_application, icsc_sub_application

def test_sc3_basic(synthetic_data, tmp_path):
    X, Y = synthetic_data
    # Use a very small subset for speed
    X_small = X[:20, :5]  # 20 cells, 5 genes
    Y_small = Y[:20]

    output_dir = tmp_path / "sc3_out"

    # Run SC3
    labels, biology_res, time_cost = sc3.sc3_application(
        X_small,
        Y=Y_small,
        nClusters=3,
        output_directory=str(output_dir),
        gene_filter=False,
        biology=False,
        n_cores=1,
        seed=2026,
        kmeans_nstart=10,
        kmeans_iter_max=100
    )

    assert labels.shape == (20,)
    assert isinstance(time_cost, float)
    assert (output_dir / "fig" / "consensus_matrix.png").exists()

@patch('pce.applications.dcc.train_representation')
@patch('pce.applications.dcc.run_consensus_clustering')
@patch('pce.applications.dcc.visualize_consensus_and_representations')
@patch('pce.applications.dcc.run_sensitivity_analysis')
def test_dcc_pipeline(mock_sensitivity, mock_viz, mock_consensus, mock_train, tmp_path):
    """
    Test the DCC application pipeline logic (mocking heavy ML parts).
    """
    input_path = str(tmp_path / "data.mat")
    output_path = str(tmp_path / "dcc_out")
    
    # Create dummy file to simulate input existence
    with open(input_path, 'w') as f:
        f.write("dummy")

    # Run DCC
    dcc_application(
        input_path=input_path,
        output_path=output_path,
        input_dim=100,
        hidden_dims=[50, 20],
        k_min=2,
        k_max=4,
        run_viz=True,
        run_sensitivity=True,
        epochs=1
    )

    # Verify calls
    # 1. train_representation should be called for each hidden_dim
    assert mock_train.call_count == 2 
    mock_train.assert_any_call(
        input_path=input_path,
        output_path=output_path,
        input_dim=100,
        hidden_dim=50,
        cuda=pytest.approx(0, abs=1),
        epochs=1
    )

    # 2. run_consensus_clustering should be called once
    mock_consensus.assert_called_once_with(
        input_path=input_path,
        output_path=output_path,
        hidden_dims=[50, 20],
        k_min=2,
        k_max=4
    )


@patch('pce.applications.dcc.train_representation')
@patch('pce.applications.dcc.run_consensus_clustering')
@patch('pce.applications.dcc.visualize_consensus_and_representations')
@patch('pce.applications.dcc.run_sensitivity_analysis')
@patch('os.path.exists')
def test_dcc_pipeline_with_results(mock_exists, mock_sensitivity, mock_viz, mock_consensus, mock_train, tmp_path):
    """
    Test DCC pipeline assuming results exist, triggering viz and sensitivity.
    """
    input_path = str(tmp_path / "data.mat")
    output_path = str(tmp_path / "dcc_out")
    
    # Force exists to True so the analysis loop proceeds
    mock_exists.return_value = True

    dcc_application(
        input_path=input_path,
        output_path=output_path,
        input_dim=100,
        hidden_dims=[50],
        k_min=2,
        k_max=2,
        run_viz=True,
        run_sensitivity=True
    )
    
    # Now these should be called
    mock_viz.assert_called()
    mock_sensitivity.assert_called()


@patch('pce.applications.icsc.single_multiple_run')
def test_icsc_mul_application(mock_worker, tmp_path):
    """
    Test ICSC Multiple Run application.
    """
    data_dir = tmp_path / "subjects"
    save_dir = tmp_path / "results"
    
    # Create dummy subject directories
    (data_dir / "subject1").mkdir(parents=True)
    (data_dir / "subject2").mkdir(parents=True)
    
    icsc_mul_application(
        num_nodes=100,
        num_threads=1,
        num_runs=2,
        dataset="test_ds",
        data_directory=str(data_dir),
        max_labels=5,
        min_labels=2,
        percent_threshold=0.1,
        save_dir=str(save_dir)
    )
    
    # num_runs=2 means we expect 2 calls to single_multiple_run
    assert mock_worker.call_count == 2
    
    # Verify arguments of the first call
    args, _ = mock_worker.call_args_list[0]
    param_tuple = args[0]
    assert param_tuple[0] == 0
    assert param_tuple[1] == str(data_dir)
    assert len(param_tuple[3]) == 2


@patch('pce.applications.icsc.single_subject_run')
@patch('pce.applications.icsc.get_threshold')
def test_icsc_sub_application(mock_threshold, mock_worker, tmp_path):
    """
    Test ICSC Subject Level application.
    """
    data_dir = tmp_path / "subjects"
    save_dir = tmp_path / "results"
    
    # Create subject structure with valid data files
    sub1 = data_dir / "subject1"
    sub1.mkdir(parents=True)
    
    # Create a dummy .npy file
    dummy_data = np.random.rand(10, 10)
    np.save(sub1 / "sess1_corr.npy", dummy_data)
    
    # Mock threshold return
    mock_threshold.return_value = 0.5
    
    icsc_sub_application(
        num_nodes=10,
        num_threads=1,
        dataset="test_ds",
        data_directory=str(data_dir),
        max_labels=5,
        min_labels=2,
        percent_threshold=0.1,
        save_dir=str(save_dir)
    )
    
    # Expect 1 call because we have 1 subject with valid data
    assert mock_worker.call_count == 1
    
    args, _ = mock_worker.call_args
    param_tuple = args[0]
    assert param_tuple[0] == 0
    assert param_tuple[2] == [0]
    assert 0 in param_tuple[3]
