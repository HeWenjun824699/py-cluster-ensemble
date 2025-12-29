import os
import pandas as pd

def sc3_export_results(analysis_results, output_dir, file_name="sc3_results.xlsx"):
    """
    Export organized results to Excel.
    R equivalent: sc3_export_results_xls
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing DataFrames. Keys will be used as sheet names.
        Example: {'Cells': cells_df, 'Genes': genes_df}
    output_dir : str
        Directory to save the file.
    file_name : str
        Name of the Excel file.
    """
    if not analysis_results:
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    path = os.path.join(output_dir, file_name)
    
    try:
        with pd.ExcelWriter(path) as writer:
            has_data = False
            
            for sheet_name, df in analysis_results.items():
                if df is not None and not df.empty:
                    # Write index (row names) as per R implementation
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
                    has_data = True
            
            if not has_data:
                # Create a dummy sheet if all are empty to avoid error
                pd.DataFrame({'Info': ['No significant results found']}).to_excel(writer, sheet_name='Info', index=False)
                
        print(f"SC3 results exported to: {path}")
        return path
    except Exception as e:
        print(f"Failed to export SC3 results: {e}")
        return None
