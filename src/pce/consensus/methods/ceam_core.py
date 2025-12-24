import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .CEAM_TKDE_2024.CEAM import CEAM


def ceam_core(BPi: np.ndarray, nCluster: int, alpha: float = 0.85, knn_size: int = 20) -> np.ndarray:
    """
    Core implementation of CEAM (Consensus Ensemble with Adaptive Metric?).

    Logic mirrors the inner loop of run_CEAM_TKDE_2024.m:
        YY = cell(1, nBase);
        Ws = cell(1, nBase);
        for iBase = 1:nBase
            YY{iBase} = ind2vec(BPi(:, iBase)')';
            Ws{iBase} = YY{iBase} * YY{iBase}';
        end
        label = CEAM(YY, Ws, nCluster, alpha, knn_size);

    Parameters
    ----------
    BPi : np.ndarray
        Base Partitions subset (n_samples, n_base).
    nCluster : int
        Target number of clusters.
    alpha : float
        Hyperparameter alpha (default 0.85).
    knn_size : int
        Size of KNN (default 20).

    Returns
    -------
    labels : np.ndarray
        Consensus cluster labels (1D array).
    """
    nSmp = BPi.shape[0]
    nBase = BPi.shape[1]

    # Initialize lists to store One-Hot matrices (YY) and Co-Association matrices (Ws)
    # MATLAB: YY = cell(1, nBase); Ws = cell(1, nBase);
    YY_list = []
    Ws_list = []

    # OneHotEncoder setup to mimic ind2vec behavior
    # categories='auto' automatically determines unique labels
    enc = OneHotEncoder(sparse_output=False, dtype=np.float32)

    for i in range(nBase):
        # Extract current base partition (column)
        # Reshape to (n_samples, 1) for OneHotEncoder
        bi = BPi[:, i].reshape(-1, 1)

        # 1. Generate One-Hot Matrix (YY)
        # MATLAB: YY{iBase} = ind2vec(BPi(:, iBase)')';
        # Note: ind2vec creates a sparse matrix where rows are classes. 
        # The trailing transpose ' makes it (nSmp x nClasses).
        yi_onehot = enc.fit_transform(bi)
        YY_list.append(yi_onehot)

        # 2. Compute Co-Association Matrix for this base partition (Ws)
        # MATLAB: Ws{iBase} = YY{iBase} * YY{iBase}';
        # This creates a block-diagonal-like matrix for the specific partition
        wi = np.dot(yi_onehot, yi_onehot.T)
        Ws_list.append(wi)

    # 3. Call the CEAM
    # MATLAB: label = CEAM(YY, Ws, nCluster, alpha, knn_size);
    # Assuming CEAM accepts the lists of matrices
    labels = CEAM(YY_list, Ws_list, nCluster, alpha, knn_size)

    return np.array(labels).flatten()
