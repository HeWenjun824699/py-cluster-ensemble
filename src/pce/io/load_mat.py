import numpy as np
import scipy.io
import h5py
from pathlib import Path
from typing import Tuple, Optional, Union


def load_mat(
        file_path: Union[str, Path],
        ensure_x_float: bool = True,
        flatten_y: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 .mat 文件中加载数据 (X) 和标签 (Y)。
    自动处理 MATLAB v7.3 (HDF5) 格式和旧版格式。

    Args:
        file_path: .mat 文件路径
        ensure_x_float: 是否强制将 X 转为 float64 (默认 True)
        flatten_y: 是否将 Y 展平为 1D 数组 (默认 True，符合 sklearn 标准)

    Returns:
        X (np.ndarray): 特征矩阵 (n_samples, n_features)
        Y (np.ndarray): 标签向量 (n_samples,)

    Raises:
        FileNotFoundError: 文件不存在
        IOError: 文件读取失败或未找到关键变量
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    X, Y = None, None

    # 常见的变量名候补
    x_keys = ['X', 'data', 'fea', 'features', 'samples']
    y_keys = ['Y', 'label', 'gnd', 'labels', 'class']

    # --- 内部函数：尝试寻找变量 ---
    def find_variable(data_dict, keys):
        for key in keys:
            if key in data_dict:
                return data_dict[key]
        return None

    try:
        # === 方案 A: 尝试 scipy.io (适用于 < v7.3) ===
        # scipy 读取时会自动处理维度的顺序，通常无需手动转置
        mat = scipy.io.loadmat(str(path))
        X = find_variable(mat, x_keys)
        Y = find_variable(mat, y_keys)

    except NotImplementedError:
        # === 方案 B: 捕获 v7.3 错误，切换到 h5py ===
        try:
            with h5py.File(path, 'r') as f:
                # 注意：h5py 读取 MATLAB v7.3 数据时，维度通常是反的 (Features, Samples)
                # 因此必须转置 .T 变回 (Samples, Features)
                raw_x = find_variable(f, x_keys)
                if raw_x is not None:
                    X = np.array(raw_x).T

                raw_y = find_variable(f, y_keys)
                if raw_y is not None:
                    Y = np.array(raw_y).T
        except Exception as e:
            raise IOError(f"Failed to read v7.3 mat file: {e}")

    except Exception as e:
        raise IOError(f"Failed to load mat file: {e}")

    # === 后处理与验证 ===

    if X is None:
        raise IOError(f"Could not find feature matrix (keys tried: {x_keys}) in {path.name}")

    # 1. 类型转换 X
    if ensure_x_float and X.dtype != np.float64:
        X = X.astype(np.float64)

    # 2. 处理 Y (如果存在)
    if Y is not None:
        # 展平 Y 为 1D 数组 (n_samples,) -> 这是 Scikit-learn 的标准
        if flatten_y:
            Y = Y.ravel()

        # 强制转换 Y 的类型 (根据你的需求，通常聚类标签是 int)
        # 如果原始数据是 float 格式的整数 (1.0, 2.0)，转为 int 安全
        # 如果是连续值回归任务，则保留 float
        if np.issubdtype(Y.dtype, np.floating) and np.all(np.mod(Y, 1) == 0):
            Y = Y.astype(np.int64)

    return X, Y
