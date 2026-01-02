import numpy as np

from .CELTA_AAAI_2021.compute_MCA_jyh import compute_MCA_jyh
from .CELTA_AAAI_2021.compute_CA_jyh import compute_CA_jyh
from .CELTA_AAAI_2021.TensorEnsemble import TensorEnsemble
from .CELTA_AAAI_2021.litekmeans import litekmeans
from .CELTA_AAAI_2021.baseline_SC import baseline_SC


def celta_core(BPi: np.ndarray, n_clusters: int, lamb: float = 0.002) -> np.ndarray:
    """
    CELTA core algorithm implementation.
    Corresponds to the core workflow in the MATLAB script:
    MCA -> CA -> TensorEnsemble -> W -> baseline_SC -> litekmeans

    Parameters
    ----------
    BPi : np.ndarray
        Base Partitions matrix, shape (n_samples, n_estimators).
        Note: It is recommended to ensure 0-based indexing before passing in,
        although it usually does not affect internal MCA/CA calculations
        unless compute_MCA_jyh explicitly relies on 1-based values.
    n_clusters : int
        Target number of clusters (k).
    lamb : float
        Regularization parameter lambda.

    Returns
    -------
    labels : np.ndarray
        Clustering label results, shape (n_samples,).
    """

    # 1. Compute Micro-cluster Association Matrix (MCA)
    # MATLAB: MCA_ML = compute_MCA_jyh(BPi);
    MCA_ML = compute_MCA_jyh(BPi)

    # 2. Compute Co-association Matrix (CA)
    # MATLAB: CA = compute_CA_jyh(BPi);
    CA = compute_CA_jyh(BPi)

    # 3. Tensor Ensemble Optimization
    # MATLAB: [A, E, B] = TensorEnsemble(MCA_ML, CA, lamb)
    # Note: Python functions typically return a tuple, automatic unpacking
    A, E, B = TensorEnsemble(MCA_ML, CA, lamb)

    # 4. Construct Similarity Matrix W
    # MATLAB: W = (A(:, :, 2) + A(:, :, 2)')/2;
    # Key point: MATLAB is 1-based indexing, A(:, :, 2) refers to the 2nd slice.
    # In Python (0-based), the corresponding index is 1.
    # Assuming the Python implementation of TensorEnsemble preserves the dimension order.
    A_slice = A[:, :, 1]
    W = (A_slice + A_slice.T) / 2

    # 5. Spectral Embedding
    # MATLAB: H_normalized = baseline_SC(W, nCluster);
    # Get normalized eigenvector matrix
    H_normalized = baseline_SC(W, n_clusters)

    # 6. K-Means Clustering
    # MATLAB: label = litekmeans(H_normalized, nCluster, 'Replicates', 10);
    # Note: The parameter name for the Python version of litekmeans may need adjustment based on implementation (usually replicates or n_init)
    label = litekmeans(H_normalized, n_clusters, replicates=10)

    # Ensure returning a 1D integer array
    return np.array(label).flatten().astype(int)
