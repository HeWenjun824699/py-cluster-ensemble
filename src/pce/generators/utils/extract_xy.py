import numpy as np
import scipy.io
import h5py


def extract_xy(file_path):
    """
    Enhanced extractXY: Supports standard .mat and v7.3 format (HDF5).
    """
    X = None
    Y = None

    # --- Attempt 1: Use scipy.io (for legacy/standard .mat) ---
    try:
        data = scipy.io.loadmat(file_path)

        # Look for X
        for key in ['X', 'data', 'fea', 'features']:
            if key in data:
                X = data[key]
                break

        # Look for Y
        for key in ['Y', 'label', 'gnd', 'labels']:
            if key in data:
                Y = data[key]
                break

        if X is not None:
            X = X.astype(float)

    except Exception as e:
        # If error includes v7.3 hint, or is NotImplementedError
        if 'v7.3' in str(e) or isinstance(e, NotImplementedError):
            print(f"Detected v7.3 mat file, switching to h5py for {file_path}...")

            # --- Attempt 2: Use h5py (for v7.3 .mat) ---
            try:
                with h5py.File(file_path, 'r') as f:
                    # Look for X
                    for key in ['X', 'data', 'fea', 'features']:
                        if key in f:
                            # [Key] MATLAB v7.3 is usually transposed after reading, requires .T
                            X = np.array(f[key]).T
                            break

                    # Look for Y
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

    # =========== [Key Modification] Unify type casting here ===========
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    if Y.dtype != np.float64:
        Y = Y.astype(np.float64)
    # =======================================================

    return X, Y
