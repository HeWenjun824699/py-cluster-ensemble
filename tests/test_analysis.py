import pandas as pd

from pce.analysis.plot import plot_2d_scatter, plot_coassociation_heatmap, plot_metric_line, plot_parameter_sensitivity


def test_plot_2d_scatter(synthetic_data, tmp_path):
    X, Y = synthetic_data
    save_path = tmp_path / "scatter.png"
    
    # Run with show=False
    plot_2d_scatter(X, Y, save_path=str(save_path), show=False, method='pca')
    assert save_path.exists()


def test_plot_coassociation_heatmap(base_partitions, synthetic_data, tmp_path):
    _, Y = synthetic_data
    save_path = tmp_path / "heatmap.png"
    
    plot_coassociation_heatmap(Y, BPs=base_partitions, save_path=str(save_path), show=False)
    assert save_path.exists()


def test_plot_metric_line(tmp_path):
    results = [
        {'ACC': 0.5, 'NMI': 0.4},
        {'ACC': 0.6, 'NMI': 0.5}
    ]
    save_path = tmp_path / "line.png"
    
    plot_metric_line(results, metrics=['ACC', 'NMI'], save_path=str(save_path), show=False)
    assert save_path.exists()


def test_plot_parameter_sensitivity(tmp_path):
    # Create dummy csv
    csv_file = tmp_path / "grid_summary.csv"
    df = pd.DataFrame({
        't': [0.1, 0.5, 0.9],
        'NMI': [0.5, 0.7, 0.6],
        'ACC': [0.5, 0.7, 0.6],
        'consensus_method': ['lwea', 'lwea', 'lwea'],
        'generator_method': ['litekmeans', 'litekmeans', 'litekmeans']
    })
    df.to_csv(csv_file, index=False)
    
    save_path = tmp_path / "sensitivity.png"
    
    plot_parameter_sensitivity(
        str(csv_file),
        target_param='t',
        metric='NMI',
        save_path=str(save_path),
        show=False
    )
    assert save_path.exists()
