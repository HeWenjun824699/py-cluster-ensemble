import os
import pandas as pd

def sc3_export_results(cells_df, genes_df, output_dir, file_name="sc3_results.xlsx"):
    """
    Export organized results to Excel.
    Matches the R SC3-Nature methods-2017 structure:
    - Sheet "Cells": Row names are cells, columns are clusters/outliers.
    - Sheet "Genes": Row names are genes, columns are DE/Marker stats.

    Parameters
    ----------
    cells_df : pd.DataFrame
        DataFrame containing cell data (Labels, Outlier Scores).
    genes_df : pd.DataFrame
        DataFrame containing gene data (DE, Markers).
    output_dir : str
        Directory to save the file.
    file_name : str
        Name of the Excel file.
    """
    if cells_df is None and genes_df is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = os.path.join(output_dir, file_name)

    try:
        with pd.ExcelWriter(path) as writer:
            if cells_df is not None:
                # R output usually includes the index (row names) if they are meaningful
                # In our case, cell_names are usually the index of cells_df
                cells_df.to_excel(writer, sheet_name='Cells', index=True)

            if genes_df is not None:
                # R output includes gene names as row names (index)
                # Ensure genes_df has gene symbols as index or include them
                # If feature_symbol is a column, we might want to set it as index for export
                # to match R's row.names=TRUE behavior strictly.
                if 'feature_symbol' in genes_df.columns:
                    genes_df_export = genes_df.set_index('feature_symbol')
                    genes_df_export.to_excel(writer, sheet_name='Genes', index=True)
                else:
                    genes_df.to_excel(writer, sheet_name='Genes', index=True)

        print(f"SC3-Nature methods-2017 results exported to: {path}")
        return path
    except Exception as e:
        print(f"Failed to export SC3-Nature methods-2017 results: {e}")
        return None
