import numpy as np

from .MDEC_TCYB_2022.getAllSegs import getAllSegs
from .MDEC_TCYB_2022.getECI import getECI
from .MDEC_TCYB_2022.getLWCA import getLWCA
from .MDEC_TCYB_2022.performHC import performHC

def mdechc_core(BPi: np.ndarray, nCluster: int, nBase: int) -> np.ndarray:
    """
    MDECHC 核心算法逻辑 (Core Logic).
    对应 MATLAB 流程:
    1. getAllSegs
    2. getECI
    3. getLWCA (计算共识关联矩阵 S)
    4. performHC (在 S 上执行层次聚类)

    Parameters
    ----------
    BPi : np.ndarray
        当前轮次的基聚类器切片 (n_samples, n_base)
    nCluster : int
        目标聚类数
    nBase : int
        基聚类器数量 (用于 getLWCA 归一化或权重计算)

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
    # 假设 getECI 默认处理或您已在内部处理 flag=1
    ECI = getECI(bcs, baseClsSegs, 1)

    # 3. 计算局部加权共识关联矩阵 (LWCA)
    # MATLAB: S = getLWCA(baseClsSegs, ECI, nBase);
    S = getLWCA(baseClsSegs, ECI, nBase)

    # 4. 执行层次聚类 (Hierarchical Clustering)
    # MATLAB: label = performHC(S, nCluster);
    label_pred = performHC(S, nCluster)

    return label_pred
