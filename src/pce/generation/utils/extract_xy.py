import numpy as np
import scipy.io
import h5py


def extract_xy(file_path):
    """
    增强版 extractXY：支持标准 .mat 和 v7.3 格式 (HDF5)。
    """
    X = None
    Y = None

    # --- 尝试 1: 使用 scipy.io (适用于旧版/标准 .mat) ---
    try:
        data = scipy.io.loadmat(file_path)

        # 寻找 X
        for key in ['X', 'data', 'fea', 'features']:
            if key in data:
                X = data[key]
                break

        # 寻找 Y
        for key in ['Y', 'label', 'gnd', 'labels']:
            if key in data:
                Y = data[key]
                break

        if X is not None:
            X = X.astype(float)

    except Exception as e:
        # 如果报错包含 v7.3 提示，或者是 NotImplementedError
        if 'v7.3' in str(e) or isinstance(e, NotImplementedError):
            print(f"Detected v7.3 mat file, switching to h5py for {file_path}...")

            # --- 尝试 2: 使用 h5py (适用于 v7.3 .mat) ---
            try:
                with h5py.File(file_path, 'r') as f:
                    # 寻找 X
                    for key in ['X', 'data', 'fea', 'features']:
                        if key in f:
                            # 【关键】MATLAB v7.3 读入后通常是转置的，需要 .T
                            X = np.array(f[key]).T
                            break

                    # 寻找 Y
                    for key in ['Y', 'label', 'gnd', 'labels']:
                        if key in f:
                            Y = np.array(f[key]).T
                            break

                if X is not None:
                    X = X.astype(float)

            except Exception as h5_e:
                print(f"h5py failed as well: {h5_e}")
                return None, None
        else:
            print(f"Error loading {file_path}: {e}")
            return None, None

    if X is None or Y is None:
        print(f"Could not find X or Y in {file_path}")
        return None, None

    # =========== 【关键修改】统一在这里强制转换类型 ===========
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    if Y.dtype != np.float64:
        Y = Y.astype(np.float64)
    # =======================================================

    return X, Y
