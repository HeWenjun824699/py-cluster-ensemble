import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import OneHotEncoder

from .SPACE_TNNLS_2024.SPACE import SPACE


def space_core(
        BPi: np.ndarray,
        nCluster: int,
        Y: np.ndarray,
        gamma: float = 4.0,
        batch_size: int = 50,
        delta: float = 0.1,
        n_active_rounds: int = 10
) -> np.ndarray:
    """
    Core implementation of SPACE (Spectral Poly-Analyte Cluster Ensemble?).

    Logic mirrors the inner loop of run_SPACE_TNNLS_2024.m:
        1. Construct Ai (3D tensor of co-association matrices).
        2. Calculate derived gamma.
        3. Call SPACE function.
        4. Derive labels using connected components on the similarity matrix S.

    Parameters
    ----------
    BPi : np.ndarray
        Base Partitions subset (n_samples, n_base).
    nCluster : int
        Target number of clusters.
    Y : np.ndarray
        True labels. The MATLAB implementation passes this to SPACE, 
        likely for semi-supervised constraints or internal NMI calculation.
    gamma : float, default=4.0
        'gam' parameter from MATLAB script.
    batch_size : int, default=50
        'batchsize' parameter.
    delta : float, default=0.1
        'delta' parameter.
    n_active_rounds : int, default=10
        Number of active learning iterations.

    Returns
    -------
    labels : np.ndarray
        Consensus cluster labels (1D array).
    """
    nSmp = BPi.shape[0]
    nBase = BPi.shape[1]

    # 1. Construction of Ai (Tensor of Co-Association Matrices)
    # MATLAB: 
    # Ai=zeros(nSmp,nSmp,nBase);
    # for j=1:nBase
    #    tmp = ind2vec(BPi(:,j)')'; YY=sparse(tmp); Ai(:,:,j)=full(YY*YY');
    # end

    # Pre-allocate Ai tensor
    Ai = np.zeros((nSmp, nSmp, nBase), dtype=np.float32)

    enc = OneHotEncoder(sparse_output=False, dtype=np.float32)

    for j in range(nBase):
        # Extract current base partition, reshape for encoder
        bj = BPi[:, j].reshape(-1, 1)

        # One-hot encoding
        yy_j = enc.fit_transform(bj)

        # Compute co-association matrix for this partition
        # YY * YY'
        ai_j = np.dot(yy_j, yy_j.T)

        # Store in tensor
        Ai[:, :, j] = ai_j

    # 2. Hyperparameter Calculation
    # MATLAB: 
    # gamma=(gam-1).*0.1; 
    # gamma=gamma.^2./nBase;
    gamma_val = (gamma - 1.0) * 0.1
    gamma_val = (gamma_val ** 2) / nBase

    # 3. Call SPACE Algorithm
    # MATLAB: [S, constraints, res_acc, res_nmi] = SPACE(Ai, nCluster, gamma, Y, batchsize, 10, delta);
    # Note: '10' appears to be an internal iteration count, hardcoded here as per MATLAB script.
    # We pass Y if available, otherwise None (depending on SPACE implementation needs).
    # Assumes SPACE returns S as the first return value.
    try:
        # Assuming the signature matches the MATLAB inputs
        S, constraints = SPACE(Ai, nCluster, gamma_val, Y, batch_size, n_active_rounds, delta)
    except TypeError:
        # Fallback if Python SPACE implementation has slightly different unpacking
        result = SPACE(Ai, nCluster, gamma_val, Y, batch_size, n_active_rounds, delta)
        S = result[0]

    # 4. Generate Labels from Similarity Matrix S
    # MATLAB: 
    # G_temp = graph(S); 
    # ypred = conncomp(G_temp);

    # Ensure S is in a format suitable for graph analysis
    # If S is dense, convert to sparse for efficiency in connected_components
    if not isinstance(S, (csr_matrix, np.ndarray)):
        S = np.array(S)

    S_sparse = csr_matrix(S)

    # Calculate connected components
    # directed=False ensures it treats the graph as undirected (symmetric)
    n_components, labels = connected_components(csgraph=S_sparse, directed=False, return_labels=True)

    return np.array(labels).flatten()
