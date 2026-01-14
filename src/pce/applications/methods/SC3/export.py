import os
import pandas as pd


def sc3_export_results(cells_df, genes_df, output_dir, file_name="sc3_results.xlsx"):
    """
    Export organized results to Excel with clean formatting and adaptive column widths.
    """
    if cells_df is None and genes_df is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = os.path.join(output_dir, file_name)

    try:
        # 指定 engine='xlsxwriter' 进行高级格式控制
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # 定义一个“纯净”格式：无粗体、无边框、左对齐
            clean_format = workbook.add_format({
                'bold': False,
                'border': 0,
                'align': 'left',
                'valign': 'vcenter'
            })

            def write_clean_sheet(df, sheet_name):
                # 1. 预处理：将 Index 变成普通列，方便统一处理
                # 如果 Index 没有名字，reset_index 会把它命名为 'index'，我们需要把它改为空
                idx_name = df.index.name if df.index.name else ""
                df_export = df.reset_index()

                # 2. 写入数据体 (Body)
                # 关键技巧：startrow=1 (留出表头行), header=False (不让pandas写表头), index=False (不让pandas写索引)
                df_export.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)

                worksheet = writer.sheets[sheet_name]

                # 3. 手动写入表头 (Header) 并计算自适应列宽
                for i, col in enumerate(df_export.columns):
                    # 处理表头文字：如果是原本的 Index 列且没名字，就设为空字符串（实现 A1 留空）
                    header_text = col
                    if i == 0 and col == "index" and idx_name == "":
                        header_text = ""
                    elif i == 0:
                        header_text = idx_name

                    # 写入表头单元格，应用 clean_format
                    worksheet.write(0, i, header_text, clean_format)

                    # --- 自适应列宽逻辑 ---
                    # 获取该列所有数据（转为字符串计算长度）
                    col_data = df_export[col].astype(str).tolist()

                    # 计算最大长度：取 (表头长度, 数据最大长度) 的最大值
                    max_len_data = max([len(x) for x in col_data]) if col_data else 0
                    max_len_header = len(str(header_text))

                    # 加上 padding (2个字符) 这里的 1.2 是经验系数，防止 Excel 渲染太紧
                    final_width = max(max_len_data, max_len_header) * 1.2
                    # 限制最大宽度，防止某些长文本把 Excel 撑爆
                    final_width = min(final_width, 50)

                    # 设置列宽
                    worksheet.set_column(i, i, final_width, clean_format)

            # --- 处理 Cells Sheet ---
            if cells_df is not None:
                # 确保索引名为空，这样左上角就是空的
                cells_df.index.name = None
                write_clean_sheet(cells_df, 'Cells')

            # --- 处理 Genes Sheet ---
            if genes_df is not None:
                if 'feature_symbol' in genes_df.columns:
                    genes_df_export = genes_df.set_index('feature_symbol')
                else:
                    genes_df_export = genes_df

                # 确保索引名为空
                genes_df_export.index.name = None
                write_clean_sheet(genes_df_export, 'Genes')

        print(f"SC3 results exported to: {path}")
        return path
    except Exception as e:
        print(f"Failed to export SC3 results: {e}")
        return None
