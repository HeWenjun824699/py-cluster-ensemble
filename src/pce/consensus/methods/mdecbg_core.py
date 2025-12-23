import numpy as np

from .MDEC_TCYB_2022.getAllSegs import getAllSegs
from .MDEC_TCYB_2022.getECI import getECI
from .MDEC_TCYB_2022.performBG import performBG

def mdecbg_core(BPi: np.ndarray, nCluster: int) -> np.ndarray:
    """
    MDECBG 核心算法逻辑 (Core Logic).

    Parameters
    ----------
    BPi : np.ndarray
        当前轮次的基聚类器切片 (n_samples, n_base)
    nCluster : int
        目标聚类数

    Returns
    -------
    label_pred : np.ndarray
        预测的聚类标签
    """
    # 1. 获取所有 Segments
    # MATLAB: [bcs, baseClsSegs] = getAllSegs(BPi);
    bcs, baseClsSegs = getAllSegs(BPi)

    # 2. 计算 ECI (Entropy-based Consensus Index)
    # MATLAB: ECI = getECI(bcs, baseClsSegs, 1);
    # 注意：如果您的 getECI Python 实现支持 flag 参数，请在此处添加，例如 getECI(bcs, baseClsSegs, 1)
    ECI = getECI(bcs, baseClsSegs, 1)

    # Note: MATLAB 代码中有 S = getLWCA(baseClsSegs, ECI, nBase);
    # 但由于 S 未被后续步骤使用，且为了效率，此处省略 getLWCA 调用。

    # 3. 执行二分图划分
    # MATLAB: label = performBG(baseClsSegs, ECI, nCluster);
    label_pred = performBG(baseClsSegs, ECI, nCluster)

    return label_pred
