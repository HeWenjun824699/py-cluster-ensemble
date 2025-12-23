import numpy as np
from typing import Optional, Tuple


from .ECCMS_TNNLS_2023.getAllSegs import getAllSegs
from .ECCMS_TNNLS_2023.computeECI import computeECI
from .ECCMS_TNNLS_2023.computeLWCA import computeLWCA
from .ECCMS_TNNLS_2023.getCA import getCA
from .ECCMS_TNNLS_2023.getHC import getHC
from .ECCMS_TNNLS_2023.run_EC_CMS import run_EC_CMS


def eccms_core(
        BPi: np.ndarray,
        n_cluster: int,
        n_base: int,
        alpha: float = 0.8,
        lamb: float = 0.4
) -> np.ndarray:
    """
    Core implementation of the ECCMS algorithm iteration.
    对应 MATLAB 脚本 run_ECCMS_TNNLS_2023.m 中的核心循环逻辑。

    Translates the following MATLAB logic:
        [bcs, baseClsSegs] = getAllSegs(BPi);
        ECI = computeECI(bcs, baseClsSegs, lambda);
        LWCA = computeLWCA(baseClsSegs, ECI, nBase);
        CA = getCA(baseClsSegs, nBase);
        A = getHC(CA, alpha);
        label = run_EC_CMS(A, LWCA, nCluster, lambda);

    Parameters
    ----------
    BPi : np.ndarray
        当前轮次的基聚类器结果 (Subset of Base Partitions), shape (n_samples, n_base).
    n_cluster : int
        目标聚类簇数 (Target number of clusters).
    n_base : int
        基聚类器数量 (Number of base estimators).
    alpha : float, default=0.8
        高置信度矩阵阈值参数 (Threshold parameter for HC matrix).
    lamb : float, default=0.4
        用于 ECI 计算和谱聚类的参数 (Parameter for ECI computation and spectral clustering).
        注意：对应 MATLAB 中的 lambda。

    Returns
    -------
    label : np.ndarray
        预测的聚类标签 (Predicted cluster labels), shape (n_samples,).
    """

    # 1. Get Segments and Basic Cluster Segments
    # MATLAB: [bcs, baseClsSegs] = getAllSegs(BPi);
    bcs, base_cls_segs = getAllSegs(BPi)

    # 2. Compute ECI (Entropy-based Consensus Information)
    # MATLAB: ECI = computeECI(bcs, baseClsSegs, lambda);
    eci = computeECI(bcs, base_cls_segs, lamb)

    # 3. Compute LWCA (Locally Weighted Co-Association)
    # MATLAB: LWCA = computeLWCA(baseClsSegs, ECI, nBase);
    lwca = computeLWCA(base_cls_segs, eci, n_base)

    # 4. Compute CA (Standard Co-Association)
    # MATLAB: CA = getCA(baseClsSegs, nBase);
    ca = getCA(base_cls_segs, n_base)

    # 5. Compute HC (High Confidence Matrix)
    # MATLAB: A = getHC(CA, alpha);
    A = getHC(ca, alpha)

    # 6. Run Final Spectral Clustering / NMF Solver
    # MATLAB: label = run_EC_CMS(A, LWCA, nCluster, lambda);
    label = run_EC_CMS(A, lwca, n_cluster, lamb)

    # 7. Post-processing
    # 确保返回的是整数类型的一维数组
    label = np.array(label, dtype=int).flatten()

    # MATLAB 代码中有一个逻辑：if min(label) == 0, label = label + 1
    # Python 通常使用 0-based 索引，因此如果你希望保持 Python 风格（0 到 k-1），可以直接返回。
    # 如果 run_EC_CMS 内部返回的是 0-based，而你需要和 MATLAB 完全一致的 1-based 输出，可以在此调整。
    # 这里默认保持 run_EC_CMS 的原始输出（通常建议 Python 保持 0-based）。

    return label
