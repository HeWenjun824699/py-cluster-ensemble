import pytest
import numpy as np
from sklearn.datasets import make_blobs


@pytest.fixture(scope="session")
def synthetic_data():
    """
    Generates a small synthetic dataset for testing.
    Returns:
        X (np.ndarray): Data matrix (n_samples, n_features).
        Y (np.ndarray): True labels (n_samples,).
    """
    X, Y = make_blobs(n_samples=100, n_features=10, centers=3, random_state=2026)
    # Ensure labels are integers
    Y = Y.astype(int)
    return X, Y


@pytest.fixture(scope="session")
def base_partitions(synthetic_data):
    """
    Generates a set of synthetic base partitions.
    Returns:
        BPs (np.ndarray): Base partitions (n_samples, n_base_partitions).
    """
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_bps = 10
    n_clusters = 3
    
    # Generate random partitions
    np.random.seed(2026)
    BPs = np.random.randint(0, n_clusters, size=(n_samples, n_bps))
    
    # Make sure at least one partition is somewhat similar to Y (optional, but realistic)
    BPs[:, 0] = Y
    
    return BPs
