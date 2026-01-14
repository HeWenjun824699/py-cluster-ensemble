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
        # Specify engine='xlsxwriter' for advanced format control
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Define a "clean" format: no bold, no border, left-aligned
            clean_format = workbook.add_format({
                'bold': False,
                'border': 0,
                'align': 'left',
                'valign': 'vcenter'
            })

            def write_clean_sheet(df, sheet_name):
                # 1. Preprocessing: Convert Index to a regular column for unified processing
                # If the Index has no name, reset_index will name it 'index', we need to set it to empty
                idx_name = df.index.name if df.index.name else ""
                df_export = df.reset_index()

                # 2. Write data body
                # Key trick: startrow=1 (reserve header row), header=False (prevent pandas from writing header), index=False (prevent pandas from writing index)
                df_export.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)

                worksheet = writer.sheets[sheet_name]

                # 3. Manually write header and calculate adaptive column width
                for i, col in enumerate(df_export.columns):
                    # Process header text: if it's the original Index column with no name, set to empty string (to leave A1 blank)
                    header_text = col
                    if i == 0 and col == "index" and idx_name == "":
                        header_text = ""
                    elif i == 0:
                        header_text = idx_name

                    # Write header cell with clean_format applied
                    worksheet.write(0, i, header_text, clean_format)

                    # --- Adaptive column width logic ---
                    # Get all data in the column (convert to string to calculate length)
                    col_data = df_export[col].astype(str).tolist()

                    # Calculate maximum length: take the maximum value of (header length, maximum data length)
                    max_len_data = max([len(x) for x in col_data]) if col_data else 0
                    max_len_header = len(str(header_text))

                    # Add padding (2 characters) - the 1.2 here is an empirical coefficient to prevent Excel from rendering too tightly
                    final_width = max(max_len_data, max_len_header) * 1.2
                    # Limit maximum width to prevent Excel from being stretched by long text
                    final_width = min(final_width, 50)

                    # Set column width
                    worksheet.set_column(i, i, final_width, clean_format)

            # --- Process Cells Sheet ---
            if cells_df is not None:
                # Ensure index name is empty so the top-left corner is blank
                cells_df.index.name = None
                write_clean_sheet(cells_df, 'Cells')

            # --- Process Genes Sheet ---
            if genes_df is not None:
                if 'feature_symbol' in genes_df.columns:
                    genes_df_export = genes_df.set_index('feature_symbol')
                else:
                    genes_df_export = genes_df

                # Ensure index name is empty
                genes_df_export.index.name = None
                write_clean_sheet(genes_df_export, 'Genes')

        print(f"SC3 results exported to: {path}")
        return path
    except Exception as e:
        print(f"Failed to export SC3 results: {e}")
        return None