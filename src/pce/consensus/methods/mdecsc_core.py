import numpy as np

from .MDEC_TCYB_2022.getAllSegs import getAllSegs
from .MDEC_TCYB_2022.getECI import getECI
from .MDEC_TCYB_2022.getLWCA import getLWCA
from .MDEC_TCYB_2022.performSC import performSC

def mdecsc_core(BPi: np.ndarray, nCluster: int, nBase: int) -> np.ndarray:
    """
    MDECSC core algorithm logic.
    Corresponding MATLAB workflow:
    1. getAllSegs
    2. getECI
    3. getLWCA (Compute consensus association matrix S)
    4. performSC (Perform spectral clustering on S)

    Parameters
    ----------
    BPi : np.ndarray
        Base partition slice for the current round (n_samples, n_base)
    nCluster : int
        Target number of clusters
    nBase : int
        Number of base estimators (used for getLWCA normalization or weight calculation)

    Returns
    -------
    label_pred : np.ndarray
        Predicted cluster labels
    """
    # 1. Get all Segments
    # MATLAB: [bcs, baseClsSegs] = getAllSegs(BPi);
    bcs, baseClsSegs = getAllSegs(BPi)

    # 2. Compute ECI (Entropy-based Consensus Index)
    # MATLAB: ECI = getECI(bcs, baseClsSegs, 1);
    ECI = getECI(bcs, baseClsSegs, 1)

    # 3. Compute Locally Weighted Co-Association matrix (LWCA)
    # MATLAB: S = getLWCA(baseClsSegs, ECI, nBase);
    S = getLWCA(baseClsSegs, ECI, nBase)

    # 4. Perform Spectral Clustering
    # MATLAB: label = performSC(S, nCluster);
    label_pred = performSC(S, nCluster)

    return label_pred
