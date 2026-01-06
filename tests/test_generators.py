import numpy as np

from pce.generators import litekmeans, cdkmeans, rskmeans, rpkmeans, bagkmeans, hetero_clustering, spectral


def test_litekmeans_basic(synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 5
    
    # Test with Y provided
    BPs = litekmeans(X=X, Y=Y, nPartitions=n_partitions)
    
    assert BPs.shape == (n_samples, n_partitions)
    assert np.all(BPs > 0)  # 1-based indexing
    assert np.issubdtype(BPs.dtype, np.floating) or np.issubdtype(BPs.dtype, np.integer)


def test_litekmeans_no_y(synthetic_data):
    X, _ = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 5
    
    # Test without Y
    BPs = litekmeans(X=X, Y=None, nPartitions=n_partitions)
    
    assert BPs.shape == (n_samples, n_partitions)


def test_cdkmeans(synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 2
    
    # Test cdkmeans
    BPs = cdkmeans(X=X, Y=Y, nPartitions=n_partitions, maxiter=10)
    
    assert BPs.shape == (n_samples, n_partitions)


def test_rskmeans(synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 2

    BPs = rskmeans(X=X, Y=Y, nPartitions=n_partitions)
    assert BPs.shape == (n_samples, n_partitions)


def test_rpkmeans(synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 2

    BPs = rpkmeans(X=X, Y=Y, nPartitions=n_partitions)
    assert BPs.shape == (n_samples, n_partitions)


def test_bagkmeans(synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 2
    
    BPs = bagkmeans(X=X, Y=Y, nPartitions=n_partitions)
    assert BPs.shape == (n_samples, n_partitions)


def test_hetero_clustering(synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 5
    
    # Test with default algorithms ('auto')
    BPs = hetero_clustering(X=X, Y=Y, nPartitions=n_partitions)
    assert BPs.shape == (n_samples, n_partitions)


def test_hetero_clustering_specific_algo(synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 2
    
    # Test with specific algorithm
    BPs = hetero_clustering(X=X, Y=Y, nPartitions=n_partitions, algorithms='kmeans')
    assert BPs.shape == (n_samples, n_partitions)

def test_spectral(synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_partitions = 2

    BPs = spectral(X=X, Y=Y, nPartitions=n_partitions)
    assert BPs.shape == (n_samples, n_partitions)
