import os
import pandas as pd

def sc3_export_results(analysis_results, output_dir, file_name="sc3_results.xlsx"):
    """
    Export organized results to Excel.
    R equivalent: sc3_export_results_xls
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing DataFrames: 'de_genes', 'marker_genes', 'outliers'.
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
            
            if 'de_genes' in analysis_results and analysis_results['de_genes'] is not None:
                if not analysis_results['de_genes'].empty:
                    analysis_results['de_genes'].to_excel(writer, sheet_name='DE Genes', index=False)
                    has_data = True
                 
            if 'marker_genes' in analysis_results and analysis_results['marker_genes'] is not None:
                if not analysis_results['marker_genes'].empty:
                    analysis_results['marker_genes'].to_excel(writer, sheet_name='Marker Genes', index=False)
                    has_data = True
                 
            if 'outliers' in analysis_results and analysis_results['outliers'] is not None:
                if not analysis_results['outliers'].empty:
                    analysis_results['outliers'].to_excel(writer, sheet_name='Outliers', index=False)
                    has_data = True
                    
            if not has_data:
                # Create a dummy sheet if all are empty to avoid error
                pd.DataFrame({'Info': ['No significant results found']}).to_excel(writer, sheet_name='Info', index=False)
                
        print(f"SC3 results exported to: {path}")
        return path
    except Exception as e:
        print(f"Failed to export SC3 results: {e}")
        return None