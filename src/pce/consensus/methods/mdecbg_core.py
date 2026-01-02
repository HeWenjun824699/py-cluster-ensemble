import numpy as np

from .MDEC_TCYB_2022.getAllSegs import getAllSegs
from .MDEC_TCYB_2022.getECI import getECI
from .MDEC_TCYB_2022.performBG import performBG

def mdecbg_core(BPi: np.ndarray, nCluster: int) -> np.ndarray:
    """
    MDECBG core algorithm logic.

    Parameters
    ----------
    BPi : np.ndarray
        Base partition slice for the current round (n_samples, n_base)
    nCluster : int
        Target number of clusters

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
    # Note: If your getECI Python implementation supports the flag parameter, please add it here, e.g., getECI(bcs, baseClsSegs, 1)
    ECI = getECI(bcs, baseClsSegs, 1)

    # Note: MATLAB code has S = getLWCA(baseClsSegs, ECI, nBase);
    # But since S is not used in subsequent steps, and for efficiency, the getLWCA call is omitted here.

    # 3. Perform bipartite graph partitioning
    # MATLAB: label = performBG(baseClsSegs, ECI, nCluster);
    label_pred = performBG(baseClsSegs, ECI, nCluster)

    return label_pred
