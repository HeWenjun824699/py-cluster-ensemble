import os
import pandas as pd
import numpy as np
from typing import List, Dict


def save_xlsx(
        data: List[Dict],
        output_path: str,
        default_name: str = "result.xlsx",
        add_summary: bool = True,
        excel_format: str = "0.0000"  # Excel 的格式字符串，对应 %.4f
):
    """
    保存为 Excel (.xlsx) 格式。
    优势：可以直接指定单元格显示格式，WPS/Excel 打开时即显示为 4 位小数，且保持数值类型。
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

        # 强制后缀名为 .xlsx (防止用户传了 .csv 路径进来)
        if not final_path.endswith('.xlsx'):
            final_path = os.path.splitext(final_path)[0] + '.xlsx'

        # --- 2. 数据处理 ---
        df = pd.DataFrame(data)
        df.index = df.index + 1

        # 强制转 float，确保是数值类型
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].astype(float)

        if add_summary and not df.empty:
            stats = pd.DataFrame({
                'Mean': df.mean(numeric_only=True),
                'Std': df.std(numeric_only=True, ddof=1)
            }).T

            # 构造空行 (Excel 中 np.nan 会显示为空白单元格)
            empty_row = pd.DataFrame(
                [[np.nan] * df.shape[1]],
                columns=df.columns,
                index=['']
            )

            df = pd.concat([df, empty_row, stats])

        # --- 3. 使用 XlsxWriter 引擎保存并设置格式 ---
        # 这一步是关键：直接操作 Excel 的格式对象
        with pd.ExcelWriter(final_path, engine='xlsxwriter') as writer:
            # 写入数据
            df.to_excel(writer, index=True, index_label="Round", sheet_name='Result')

            # 获取 workbook 和 worksheet 对象
            workbook = writer.book
            worksheet = writer.sheets['Result']

            # 定义数字格式 (例如 "0.0000")
            num_fmt = workbook.add_format({'num_format': excel_format})

            # 定义表头格式 (可选：加粗、居中)
            # header_fmt = workbook.add_format({'bold': True, 'align': 'center'})

            # 智能设置列宽和格式
            # enumerate(df.columns) 拿到的是数据列名
            # 在 Excel 中，第 0 列是 Index ("Round")，数据列从第 1 列开始
            for i, col in enumerate(df.columns):
                # 如果该列是数值列，应用格式
                if col in numeric_cols:
                    # set_column(first_col, last_col, width, cell_format)
                    # i + 1 是因为第 0 列被 index 占用了
                    worksheet.set_column(i + 1, i + 1, 12, num_fmt)
                else:
                    # 非数值列，只设置宽度
                    worksheet.set_column(i + 1, i + 1, 12)

        print(f"Results saved to {final_path}")

    except Exception as e:
        print(f"Failed to save xlsx: {e}")
