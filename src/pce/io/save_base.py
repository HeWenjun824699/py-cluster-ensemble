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
    保存基聚类矩阵 (BPs) 和 真实标签 (Y) 到 .mat 文件。

    参数:
        BPs (np.ndarray): 基聚类矩阵 (N样本 x M成员) -> 保存为变量 'members'
        Y (np.ndarray): 真实标签 (N样本 x 1) -> 保存为变量 'Y'
        output_path (str): 保存路径
        default_name (str): 默认文件名，默认为 "base.mat"
    """
    try:
        # --- 1. 路径处理 ---
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

        # --- 2. 数据格式规范化 ---
        if not isinstance(BPs, np.ndarray):
            BPs = np.array(BPs)

        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)

        # 强制转换为 double (np.float64)
        Y = Y.astype(np.float64)

        # 【关键修改】强制 Y 为 N x 1 的列向量
        # 即使传入的是 (N,) 或 (1, N)，这都会将其转为 (N, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        elif Y.shape[0] == 1 and Y.shape[1] > 1:
            # 如果是 1 x N，转置为 N x 1
            Y = Y.T

        # 双重保险：检查 BPs 和 Y 的行数是否一致（N样本数应相同）
        if BPs.shape[0] != Y.shape[0]:
            print(f"Warning: Sample count mismatch! BPs: {BPs.shape[0]}, Y: {Y.shape[0]}")

        # --- 3. 构造保存字典 ---
        save_dict = {
            'BPs': BPs,  # N x M
            'Y': Y  # N x 1 (现在确保是列向量了)
        }

        # --- 4. 保存 ---
        scipy.io.savemat(final_path, save_dict)
        print(f"Base clusterings saved to {final_path}")

    except Exception as e:
        print(f"Failed to save base mat: {e}")
