import os
import pandas as pd
import numpy as np
from typing import List, Dict

import scipy


def save_mat(
        data: List[Dict],
        output_path: str,
        default_name: str = "result.mat",
        add_summary: bool = True
):
    """
    保存为 .mat 格式。
    """
    try:
        # 1. 路径处理
        if output_path.endswith(('/', '\\')) or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(output_path, default_name)
        else:
            final_path = output_path
            parent_dir = os.path.dirname(final_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        # 强制后缀名为 .mat
        if not final_path.endswith('.mat'):
            final_path = os.path.splitext(final_path)[0] + '.mat'

        # 2. 数据处理
        results_list = []
        for item in data:
            acc, nmi, purity, AR, RI, MI, HI, fscore, precision, recall, entropy, SDCS, RME, bal = item.values()
            results_list.append([acc, nmi, purity, AR, RI, MI, HI, fscore, precision, recall, entropy, SDCS, RME, bal])

        # 3. 汇总与保存
        results_mat = np.array(results_list)

        if add_summary:
            # 计算均值和方差 (注意标准差使用 ddof=1 以匹配 MATLAB std)
            summary_mean = np.mean(results_mat, axis=0)
            summary_std = np.std(results_mat, axis=0, ddof=1)

            # 构造保存字典
            save_dict = {
                'result': results_mat,
                'result_summary': summary_mean,
                'result_summary_std': summary_std
            }
        else:
            save_dict = {
                'result': results_mat
            }

        scipy.io.savemat(final_path, save_dict)
        print(f"Results saved to {final_path}")

    except Exception as e:
        print(f"Failed to save csv: {e}")
