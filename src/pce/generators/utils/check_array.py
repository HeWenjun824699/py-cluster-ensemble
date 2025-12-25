import numpy as np
import scipy.sparse as sp


def check_array(X, dtype=np.float64, accept_sparse=False):
    """
    统一处理输入数据 X：
    1. 如果算法不支持稀疏矩阵 (accept_sparse=False)，则自动转为 dense numpy array。
    2. 确保数据类型正确。
    """
    # 如果是稀疏矩阵，且该算法不接受稀疏矩阵，则转换
    if sp.issparse(X):
        if not accept_sparse:
            # 这是一个内存敏感的操作，对于超大数据集可能需要由用户决定，
            # 但对于通用工具库，为了易用性，通常默认转为 dense
            X = X.toarray()

    # 确保是 numpy 数组 (如果原本是 list 或 tuple)
    if not isinstance(X, (np.ndarray, sp.spmatrix)):
        X = np.array(X)

    return X.astype(dtype, copy=False)
