import numpy as np
import scipy.io
import h5py
from pathlib import Path
from typing import Tuple, Optional, Union, List, Any


# =========================================================================
#  Private Core Functions - Responsible for IO and MATLAB Version Compatibility
# =========================================================================

def _load_mat_core(file_path: Union[str, Path], target_keys: List[str]) -> Any:
    """
    (Internal) Attempts to read any one of the specified variable names list from the .mat file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Internal helper function: find key in dictionary/h5 object
    def find_variable(data_obj, keys):
        for key in keys:
            if key in data_obj:
                return data_obj[key]
        return None

    try:
        # 1. Try scipy.io (< v7.3)
        mat = scipy.io.loadmat(str(path))
        data = find_variable(mat, target_keys)
        if data is not None:
            return data

    except NotImplementedError:
        # 2. Try h5py (>= v7.3)
        try:
            with h5py.File(path, 'r') as f:
                data = find_variable(f, target_keys)
                if data is not None:
                    # h5py reading is transposed, needs transposition
                    return np.array(data).T
        except Exception as e:
            raise IOError(f"Failed to read v7.3 mat file: {e}")
    except Exception as e:
        raise IOError(f"Failed to load mat file: {e}")

    return None


# =========================================================================
#  Public APIs
# =========================================================================

def load_mat_X_Y(
        file_path: Union[str, Path],
        ensure_x_float: bool = True,
        flatten_y: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw feature data (X) and labels (Y).
    """
    # Define variable names to search for
    x_keys = ['X', 'data', 'fea', 'features', 'samples']
    y_keys = ['Y', 'label', 'gnd', 'labels', 'class']

    # Call core loading logic
    X = _load_mat_core(file_path, x_keys)
    Y = _load_mat_core(file_path, y_keys)

    # Verify if X exists
    if X is None:
        raise IOError(f"Could not find feature matrix (keys tried: {x_keys}) in {file_path}")

    # Verify if Y exists (some unsupervised scenarios allow Y to be empty, decide whether to error based on your needs)
    if Y is None:
        raise IOError(f"Could not find label vector (keys tried: {y_keys}) in {file_path}")

    # --- Post-process X ---
    if ensure_x_float and X.dtype != np.float64:
        X = X.astype(np.float64)

    # --- Post-process Y ---
    if flatten_y:
        Y = Y.ravel()

    # Smartly convert Y type (float int -> int)
    if np.issubdtype(Y.dtype, np.floating) and np.all(np.mod(Y, 1) == 0):
        Y = Y.astype(np.int64)

    return X, Y


def load_mat_BPs_Y(
        file_path: Union[str, Path],
        fix_matlab_index: bool = True,
        flatten_y: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Base Partitions (BPs) and labels (Y).

    Args:
        fix_matlab_index: If minimum value is detected as 1, automatically subtract 1 to adapt to Python (default True)
    """
    bps_keys = ['BPs', 'base_partitions', 'members', 'labels_mat']
    y_keys = ['Y', 'label', 'gnd', 'labels', 'class']

    BPs = _load_mat_core(file_path, bps_keys)
    Y = _load_mat_core(file_path, y_keys)

    if BPs is None:
        raise IOError(f"Could not find Base Partitions (keys tried: {bps_keys}) in {file_path}")

    if Y is None:
        raise IOError(f"Could not find label vector in {file_path}")

    # --- Post-process BPs ---
    # 1. Ensure integer
    if np.issubdtype(BPs.dtype, np.floating):
        BPs = BPs.astype(np.int64)

    # 2. Handle MATLAB 1-based indexing
    if fix_matlab_index and np.min(BPs) == 1:
        # This is a great automatic handling feature
        BPs = BPs - 1

    # --- Post-process Y ---
    if flatten_y:
        Y = Y.ravel()

    if np.issubdtype(Y.dtype, np.floating) and np.all(np.mod(Y, 1) == 0):
        Y = Y.astype(np.int64)

    return BPs, Y
