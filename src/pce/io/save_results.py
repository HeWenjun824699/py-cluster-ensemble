import os
import traceback
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import scipy


def save_results_csv(
        data: Union[List[Dict], Dict],
        output_path: str,
        default_name: str = "result.csv",
        add_summary: bool = True,
        float_format: str = "%.4f"
):
    """
    Save experiment results to a CSV file with optional statistical summaries.

    This function smartly handles both single-row (Dict) and multi-row (List[Dict])
    data. It supports automatic appending of mean and standard deviation rows,
    separated by an empty line for better readability.

    Parameters
    ----------
    data : Union[List[Dict], Dict]
        The evaluation results to be saved. If a single dictionary is passed,
        it is automatically wrapped into a list.
    output_path : str
        Target output path. If it refers to a directory, the file is saved
        using `default_name`. If it is a file path, the parent directories
        will be created automatically.
    default_name : str, default="result.csv"
        The filename used when `output_path` is identified as a directory.
    add_summary : bool, default=True
        If True, calculates and appends the 'Mean' and 'Std' rows. It also
        generates a formatted string row ("Mean±Std") where metric values
        are multiplied by 100.
    float_format : str, default="%.4f"
        Formatting string for floating-point numbers in the output file.

    Returns
    -------
    None
    """

    try:
        # --- Input compatibility handling (New) ---
        # If the user passed only a dictionary, automatically wrap it in a list
        if isinstance(data, dict):
            data = [data]

        # --- 1. Path handling (Unchanged) ---
        if output_path.endswith(('/', '\\')) or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(output_path, default_name)
        else:
            final_path = output_path
            parent_dir = os.path.dirname(final_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        # --- 2. Data processing ---
        df = pd.DataFrame(data)
        df.index = df.index + 1

        if add_summary and not df.empty:
            # 2.1 Calculate statistics first
            means = df.mean(numeric_only=True)
            stds = df.std(numeric_only=True, ddof=1).fillna(0)

            stats = pd.DataFrame({'Mean': means, 'Std': stds}).T

            # ================= [New Logic Start] =================
            # 2.2 Construct Str row: format as "Mean±Std" (value*100 then keep 2 decimal places)
            str_data = {}
            for col in df.columns:
                if col in means.index:
                    # [Modification] Check if column name contains 'time' (case insensitive)
                    if 'time' in str(col).lower():
                        # Time column: do not multiply by 100, keep 2 decimal places
                        m_val = means[col]
                        s_val = stds[col]
                        str_data[col] = f"{m_val:.2f}±{s_val:.2f}"
                    else:
                        # Metric column: multiply by 100, keep 2 decimal places (percentage format)
                        m_val = means[col] * 100
                        s_val = stds[col] * 100
                        str_data[col] = f"{m_val:.2f}±{s_val:.2f}"
                else:
                    str_data[col] = ""
            str_row_df = pd.DataFrame([str_data], index=['Str'])

            # 2.3 [Key Fix] Safe formatting function
            # If x is not a number (e.g. None or string), return x directly to avoid error
            def safe_format(x):
                if isinstance(x, (int, float)) and not pd.isna(x):
                    return float_format % x
                return x

            # Use map instead of applymap (Pandas 2.1+ recommendation, older versions are also fine)
            df_str = df.map(safe_format)
            stats_str = stats.map(safe_format)

            # 2.4 Construct empty row
            empty_row = pd.DataFrame(
                [[''] * df.shape[1]],
                columns=df.columns,
                index=['']
            )

            # 2.5 Concatenate
            df_final = pd.concat([df_str, empty_row, stats_str, str_row_df])

        else:
            df_final = df

        # --- 3. Save ---
        # na_rep='' ensures None is saved as empty string instead of 'NaN'
        df_final.to_csv(
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
        data: Union[List[Dict], Dict],
        output_path: str,
        default_name: str = "result.xlsx",
        add_summary: bool = True,
        excel_format: str = "0.0000"  # Excel format string, corresponds to %.4f
):
    """
    Save experiment results to an Excel (.xlsx) file with numeric type preservation.

    Compared to CSV, this method allows specifying the display format while
    keeping the cells as numeric types, which is useful for further
    calculations in spreadsheet software.

    Parameters
    ----------
    data : Union[List[Dict], Dict]
        The evaluation results to be saved.
    output_path : str
        Target output path. Parent directories are created if they do not exist.
    default_name : str, default="result.xlsx"
        Filename used if `output_path` is a directory.
    add_summary : bool, default=True
        Whether to calculate and append 'Mean', 'Std', and formatted 'Mean±Std' rows.
    excel_format : str, default="0.0000"
        Excel number format string (e.g., "0.0000" for 4 decimal places).

    Returns
    -------
    None
    """

    try:
        # --- [New] Input compatibility handling ---
        # If the user passed only a dictionary, automatically wrap it in a list
        if isinstance(data, dict):
            data = [data]

        # --- 1. Path handling ---
        if output_path.endswith(('/', '\\')) or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(output_path, default_name)
        else:
            final_path = output_path
            parent_dir = os.path.dirname(final_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        # Force extension to be .xlsx (prevent user from passing .csv path)
        if not final_path.endswith('.xlsx'):
            final_path = os.path.splitext(final_path)[0] + '.xlsx'

        # --- 2. Data processing ---
        df = pd.DataFrame(data)
        df.index = df.index + 1

        # Force convert to float, ensure numeric type
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].astype(float)

        if add_summary and not df.empty:
            means = df.mean(numeric_only=True)
            stds = df.std(numeric_only=True, ddof=1).fillna(0)

            stats = pd.DataFrame({'Mean': means, 'Std': stds}).T

            # ================= [New Logic Start] =================
            # Construct Str row: format as "Mean±Std" (value*100 then keep 2 decimal places)
            str_data = {}
            for col in df.columns:
                if col in means.index:
                    # [Modification] Check if column name contains 'time'
                    if 'time' in str(col).lower():
                        # Time column: do not multiply by 100, keep 2 decimal places
                        m_val = means[col]
                        s_val = stds[col]
                        str_data[col] = f"{m_val:.2f}±{s_val:.2f}"
                    else:
                        # Metric column: multiply by 100, keep 2 decimal places (percentage format)
                        m_val = means[col] * 100
                        s_val = stds[col] * 100
                        str_data[col] = f"{m_val:.2f}±{s_val:.2f}"
                else:
                    str_data[col] = ""

            str_row_df = pd.DataFrame([str_data], index=['Str'])
            # ================= [New Logic End] =================

            # Construct empty row
            empty_row = pd.DataFrame(
                [[np.nan] * df.shape[1]],
                columns=df.columns,
                index=['']
            )

            # Concatenate: Raw data -> Empty row -> Statistics -> Str row
            # Note: After concatenation, because of the Str row, these columns will become object type in Pandas
            # But it does not affect xlsxwriter writing the floats as numbers
            df = pd.concat([df, empty_row, stats, str_row_df])

        # --- 3. Use XlsxWriter engine to save and set format ---
        # This step is key: directly manipulate Excel format objects
        with pd.ExcelWriter(final_path, engine='xlsxwriter') as writer:
            # Write data
            df.to_excel(writer, index=True, index_label="Round", sheet_name='Result')

            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Result']

            # Define number format (e.g. "0.0000")
            num_fmt = workbook.add_format({'num_format': excel_format})

            # Define header format (optional: bold, center)
            # header_fmt = workbook.add_format({'bold': True, 'align': 'center'})

            # Smartly set column width and format
            # enumerate(df.columns) gets data column names
            # In Excel, column 0 is Index ("Round"), data columns start from column 1
            for i, col in enumerate(df.columns):
                # If the column is numeric, apply format
                if col in numeric_cols:
                    # set_column(first_col, last_col, width, cell_format)
                    # i + 1 is because column 0 is occupied by index
                    worksheet.set_column(i + 1, i + 1, 12, num_fmt)
                else:
                    # Non-numeric column, only set width
                    worksheet.set_column(i + 1, i + 1, 12)

        # print(f"Results saved to {final_path}")

    except Exception as e:
        print(f"Failed to save xlsx: {e}")


def save_results_mat(
        data: Union[List[Dict], Dict],
        output_path: str,
        default_name: str = "result.mat",
        add_summary: bool = True
):
    """
    Save experiment results to a MATLAB-compatible .mat file.

    Stores the raw result matrix and optionally saves statistical summaries
    as separate variables in the MATLAB workspace.

    Parameters
    ----------
    data : Union[List[Dict], Dict]
        The evaluation results to be saved.
    output_path : str
        Target output path.
    default_name : str, default="result.mat"
        Filename used if `output_path` is a directory.
    add_summary : bool, default=True
        If True, includes `result_summary` (Mean), `result_summary_std` (Std),
        and `result_summary_str` (Cell array of "Mean±Std" strings) in the file.

    Returns
    -------
    None
    """

    try:
        # --- 1. Input compatibility handling (New) ---
        if isinstance(data, dict):
            data = [data]

        # --- 2. Path handling ---
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

        # --- 3. Data processing (Upgraded) ---
        # [Modification 2] Use DataFrame to process data, safer than item.values()
        # It can automatically turn None (like Time) into NaN, ensuring matrix is numeric type
        df = pd.DataFrame(data)

        # Convert to numpy float matrix
        # If Time is None, it will become np.nan here, no error
        results_mat = df.astype(float).values

        # Construct basic save dictionary
        save_dict = {'result': results_mat}

        if add_summary:
            # --- 4. Statistical calculation (Fix NaN issue) ---
            # [Modification 3] Use nanmean / nanstd to ignore NaN effects
            summary_mean = np.nanmean(results_mat, axis=0)

            # Calculate standard deviation (ddof=1)
            summary_std_raw = np.nanstd(results_mat, axis=0, ddof=1)
            # [Core Fix] Replace NaN (generated by single row data) with 0.0
            summary_std = np.nan_to_num(summary_std_raw, nan=0.0)

            # --- 5. Construct string list ---
            summary_str_list = []
            for i, (m, s) in enumerate(zip(summary_mean, summary_std)):
                # Last column (Time) do not multiply by 100
                if i == len(summary_mean) - 1:
                    val_str = f"{m:.2f}±{s:.2f}"
                else:
                    val_str = f"{m * 100:.2f}±{s * 100:.2f}"
                summary_str_list.append(val_str)

            # Convert to numpy object array (corresponds to MATLAB Cell Array)
            summary_str_arr = np.array(summary_str_list, dtype=object)

            # Update dictionary
            save_dict.update({
                'result_summary': summary_mean,
                'result_summary_std': summary_std,
                'result_summary_str': summary_str_arr
            })

        # --- 6. Save ---
        scipy.io.savemat(final_path, save_dict)
        # print(f"Results saved to {final_path}")

    except Exception as e:
        traceback.print_exc()
        print(f"Failed to save mat: {e}")
