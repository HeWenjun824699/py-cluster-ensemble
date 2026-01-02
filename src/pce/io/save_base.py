import os
import numpy as np
import scipy.io


def save_base_mat(
        BPs: np.ndarray,
        Y: np.ndarray,
        output_path: str,
        default_name: str = "base.mat"
):
    """
    Save Base Partitions (BPs) and true labels (Y) to a MATLAB compatible .mat file.

    This function prepares ensemble data for cross-platform use by ensuring
    numerical compatibility with MATLAB. It automatically handles path
    resolution and data type casting.

    Parameters
    ----------
    BPs : np.ndarray
        Base Partitions matrix of shape (n_samples, n_estimators).
        The matrix is saved under the variable name 'BPs' in the .mat file.
    Y : np.ndarray
        True labels for the samples. The function performs the following
        normalizations:

        1. Casts to `np.float64` (MATLAB double).
        2. Reshapes into an $(n\_samples, 1)$ column vector.

    output_path : str
        The destination path.

        - If it points to an existing directory or ends with a slash,
          the file is saved as `default_name` within that directory.
        - If it points to a file path, it saves directly to that location,
          automatically appending '.mat' if missing.

    default_name : str, default="base.mat"
        The filename used only when `output_path` is identified as a
        directory.

    Returns
    -------
    None
    """

    try:
        # --- 1. Path handling ---
        if output_path.endswith(('/', '\\')) or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(output_path, default_name)
        else:
            final_path = output_path
            parent_dir = os.path.dirname(final_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        if not final_path.endswith('.mat'):
            final_path = os.path.splitext(final_path)[0] + '.mat'

        # --- 2. Data format normalization ---
        if not isinstance(BPs, np.ndarray):
            BPs = np.array(BPs)

        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)

        # Force convert to double (np.float64)
        Y = Y.astype(np.float64)

        # [Key Modification] Force Y to be an N x 1 column vector
        # Even if passed as (N,) or (1, N), it will be reshaped to (N, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        elif Y.shape[0] == 1 and Y.shape[1] > 1:
            # If 1 x N, transpose to N x 1
            Y = Y.T

        # Double check: Check if row counts of BPs and Y match (N samples should be the same)
        if BPs.shape[0] != Y.shape[0]:
            print(f"Warning: Sample count mismatch! BPs: {BPs.shape[0]}, Y: {Y.shape[0]}")

        # --- 3. Construct save dictionary ---
        save_dict = {
            'BPs': BPs,  # N x M
            'Y': Y  # N x 1 (now ensured to be a column vector)
        }

        # --- 4. Save ---
        scipy.io.savemat(final_path, save_dict)
        # print(f"Base clusterings saved to {final_path}")

    except Exception as e:
        print(f"Failed to save base mat: {e}")
