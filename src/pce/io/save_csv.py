import os
import pandas as pd
from typing import List, Dict


def save_csv(data: List[Dict], output_path: str, default_name="result.csv"):
    """
    智能保存 CSV。

    Args:
        output_path: 可以是文件夹路径 (例如 'results/')，也可以是具体文件路径 ('results/data.csv')
    """
    try:
        # 1. 如果路径是以 / 或 \ 结尾，或者是一个已存在的目录 -> 视为目录
        if output_path.endswith(('/', '\\')) or os.path.isdir(output_path):
            # 确保目录存在
            os.makedirs(output_path, exist_ok=True)
            # 拼接默认文件名
            final_path = os.path.join(output_path, default_name)
        else:
            # 2. 视为文件路径
            final_path = output_path
            # 确保父目录存在 (防止报错 FileNotFoundError)
            parent_dir = os.path.dirname(final_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        df = pd.DataFrame(data)
        df.to_csv(final_path, index=False)
        print(f"Results saved to {final_path}")

    except Exception as e:
        print(f"Failed to save csv: {e}")
