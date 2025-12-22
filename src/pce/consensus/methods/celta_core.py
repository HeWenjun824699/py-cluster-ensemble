import numpy as np

from .CELTA_AAAI_2021.compute_MCA_jyh import compute_MCA_jyh
from .CELTA_AAAI_2021.compute_CA_jyh import compute_CA_jyh
from .CELTA_AAAI_2021.TensorEnsemble import TensorEnsemble
from .CELTA_AAAI_2021.litekmeans import litekmeans
from .CELTA_AAAI_2021.baseline_SC import baseline_SC


def celta_core(BPi: np.ndarray, n_clusters: int, lamb: float = 0.002) -> np.ndarray:
    """
    CELTA 核心算法实现
    对应 MATLAB 脚本中的核心流程：
    MCA -> CA -> TensorEnsemble -> W -> baseline_SC -> litekmeans

    Parameters
    ----------
    BPi : np.ndarray
        基聚类矩阵 (Base Partitions), shape (n_samples, n_estimators)
        注意：传入前建议确保是 0-based 索引，但在计算 MCA/CA 内部通常不影响，
        除非 compute_MCA_jyh 内部显式依赖 1-based 数值。
    n_clusters : int
        目标聚类数 (k)
    lamb : float
        正则化参数 lambda

    Returns
    -------
    labels : np.ndarray
        聚类标签结果, shape (n_samples,)
    """

    # 1. 计算微簇关联矩阵 (MCA)
    # MATLAB: MCA_ML = compute_MCA_jyh(BPi);
    MCA_ML = compute_MCA_jyh(BPi)

    # 2. 计算共协矩阵 (CA)
    # MATLAB: CA = compute_CA_jyh(BPi);
    CA = compute_CA_jyh(BPi)

    # 3. 张量集成优化
    # MATLAB: [A, E, B] = TensorEnsemble(MCA_ML, CA, lambda);
    # 注意：Python 函数通常返回 tuple，自动解包
    A, E, B = TensorEnsemble(MCA_ML, CA, lamb)

    # 4. 构建相似度矩阵 W
    # MATLAB: W = (A(:, :, 2) + A(:, :, 2)')/2;
    # 关键点：MATLAB 是 1-based 索引，A(:, :, 2) 指的是第 2 个切片。
    # 在 Python (0-based) 中，对应索引为 1。
    # 假设 TensorEnsemble 的 Python 实现保留了维度顺序。
    A_slice = A[:, :, 1]
    W = (A_slice + A_slice.T) / 2

    # 5. 谱嵌入 (Spectral Embedding)
    # MATLAB: H_normalized = baseline_SC(W, nCluster);
    # 获取归一化的特征向量矩阵
    H_normalized = baseline_SC(W, n_clusters)

    # 6. K-Means 聚类
    # MATLAB: label = litekmeans(H_normalized, nCluster, 'Replicates', 10);
    # 注意：Python 版 litekmeans 的参数名可能需要根据实际实现调整（通常是 replicates 或 n_init）
    label = litekmeans(H_normalized, n_clusters, replicates=10)

    # 确保返回的是一维整数数组
    return np.array(label).flatten().astype(int)
