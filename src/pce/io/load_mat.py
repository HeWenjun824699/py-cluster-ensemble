import numpy as np
import scipy.io
import h5py
from pathlib import Path
from typing import Tuple, Optional, Union, List, Any


# =========================================================================
#  内部核心函数 (Private Core) - 负责处理 IO 和 MATLAB 版本兼容
# =========================================================================

def _load_mat_core(file_path: Union[str, Path], target_keys: List[str]) -> Any:
    """
    (Internal) 尝试从 .mat 文件中读取指定的变量名列表中的任意一个。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # 内部帮助函数：在字典/h5对象中查找键
    def find_variable(data_obj, keys):
        for key in keys:
            if key in data_obj:
                return data_obj[key]
        return None

    try:
        # 1. 尝试 scipy.io (< v7.3)
        mat = scipy.io.loadmat(str(path))
        data = find_variable(mat, target_keys)
        if data is not None:
            return data

    except NotImplementedError:
        # 2. 尝试 h5py (>= v7.3)
        try:
            with h5py.File(path, 'r') as f:
                data = find_variable(f, target_keys)
                if data is not None:
                    # h5py 读取也是反的，需要转置
                    return np.array(data).T
        except Exception as e:
            raise IOError(f"Failed to read v7.3 mat file: {e}")
    except Exception as e:
        raise IOError(f"Failed to load mat file: {e}")

    return None


# =========================================================================
#  公开接口 (Public APIs)
# =========================================================================

def load_mat_X_Y(
        file_path: Union[str, Path],
        ensure_x_float: bool = True,
        flatten_y: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载原始特征数据 (X) 和标签 (Y)。
    """
    # 定义搜索的变量名
    x_keys = ['X', 'data', 'fea', 'features', 'samples']
    y_keys = ['Y', 'label', 'gnd', 'labels', 'class']

    # 调用核心加载逻辑
    X = _load_mat_core(file_path, x_keys)
    Y = _load_mat_core(file_path, y_keys)

    # 验证 X 是否存在
    if X is None:
        raise IOError(f"Could not find feature matrix (keys tried: {x_keys}) in {file_path}")

    # 验证 Y 是否存在 (部分无监督场景允许 Y 为空，根据你的需求决定是否报错)
    if Y is None:
        raise IOError(f"Could not find label vector (keys tried: {y_keys}) in {file_path}")

    # --- 后处理 X ---
    if ensure_x_float and X.dtype != np.float64:
        X = X.astype(np.float64)

    # --- 后处理 Y ---
    if flatten_y:
        Y = Y.ravel()

    # 智能转换 Y 的类型 (float int -> int)
    if np.issubdtype(Y.dtype, np.floating) and np.all(np.mod(Y, 1) == 0):
        Y = Y.astype(np.int64)

    return X, Y


def load_mat_BPs_Y(
        file_path: Union[str, Path],
        fix_matlab_index: bool = True,
        flatten_y: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载基聚类结果 (BPs) 和标签 (Y)。

    Args:
        fix_matlab_index: 如果检测到最小值为 1，是否自动减 1 以适应 Python (默认 True)
    """
    bps_keys = ['BPs', 'base_partitions', 'members', 'labels_mat']
    y_keys = ['Y', 'label', 'gnd', 'labels', 'class']

    BPs = _load_mat_core(file_path, bps_keys)
    Y = _load_mat_core(file_path, y_keys)

    if BPs is None:
        raise IOError(f"Could not find Base Partitions (keys tried: {bps_keys}) in {file_path}")

    if Y is None:
        raise IOError(f"Could not find label vector in {file_path}")

    # --- 后处理 BPs ---
    # 1. 确保是整数
    if np.issubdtype(BPs.dtype, np.floating):
        BPs = BPs.astype(np.int64)

    # 2. 处理 MATLAB 1-based 索引
    if fix_matlab_index and np.min(BPs) == 1:
        # 这是一个很棒的自动处理特性
        BPs = BPs - 1

    # --- 后处理 Y ---
    if flatten_y:
        Y = Y.ravel()

    if np.issubdtype(Y.dtype, np.floating) and np.all(np.mod(Y, 1) == 0):
        Y = Y.astype(np.int64)

    return BPs, Y
