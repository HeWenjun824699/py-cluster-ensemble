import os
import pandas as pd
import numpy as np
from typing import List, Dict

import scipy


def save_results_csv(
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
                [[np.nan] * df.shape[1]],
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

        # print(f"Results saved to {final_path}")

    except Exception as e:
        print(f"Failed to save csv: {e}")


def save_results_xlsx(
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


def save_results_mat(
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
            acc, nmi, purity, AR, RI, MI, HI, fscore, precision, recall, entropy, SDCS, RME, bal, time = item.values()
            results_list.append([acc, nmi, purity, AR, RI, MI, HI, fscore, precision, recall, entropy, SDCS, RME, bal, time])

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
