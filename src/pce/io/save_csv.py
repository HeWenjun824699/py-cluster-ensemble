import os
import pandas as pd
from typing import List, Dict


def save_csv(
        data: List[Dict],
        output_path: str,
        default_name: str = "result.csv",
        add_summary: bool = True,
        float_format: str = "%.4f"
):
    """
    智能保存 CSV，支持自动追加均值和标准差，并用空行分隔。
    """
    try:
        # --- 1. 路径处理 (保持不变) ---
        if output_path.endswith(('/', '\\')) or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(output_path, default_name)
        else:
            final_path = output_path
            parent_dir = os.path.dirname(final_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        # --- 2. 数据处理 ---
        df = pd.DataFrame(data)
        df.index = df.index + 1

        if add_summary and not df.empty:
            # 2.1 计算统计量
            stats = pd.DataFrame({
                'Mean': df.mean(numeric_only=True),
                'Std': df.std(numeric_only=True, ddof=1)
            }).T

            # 2.2 【关键修改】构造一个空行
            # values 使用 None，index 使用空字符串 ''，这样在 CSV 里这行就是全空的
            empty_row = pd.DataFrame(
                [[None] * df.shape[1]],
                columns=df.columns,
                index=['']
            )

            # 2.3 拼接：原始数据 -> 空行 -> 统计数据
            df = pd.concat([df, empty_row, stats])

        # --- 3. 保存 ---
        # na_rep='' 确保 None 被保存为空字符串而不是 'NaN'
        df.to_csv(
            final_path,
            index=True,
            index_label="Round",
            float_format=float_format,
            na_rep=''
        )

        print(f"Results saved to {final_path}")

    except Exception as e:
        print(f"Failed to save csv: {e}")
