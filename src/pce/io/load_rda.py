import numpy as np
import pyreadr

def load_rda_X_Y(rda_path):
    """
    Load the standard SC3-Nature methods-2017 demo dataset 'yan.rda'.
    
    This function bridges the gap between R's data format and Python's input requirements.
    It requires the 'pyreadr' library to be installed.
    
    Parameters
    ----------
    rda_path : str
        Path to the 'yan.rda' file.
        
    Returns
    -------
    tuple
        (X, Y, gene_names, cell_names)
        - X: (n_cells, n_genes) Log-transformed expression matrix.
        - Y: (n_cells,) Integer labels derived from 'ann'.
        - gene_names: (n_genes,) Gene symbols.
        - cell_names: (n_cells,) Cell IDs.
    """

    # 1. Read the R data file
    # pyreadr returns a dictionary of dataframes
    result = pyreadr.read_r(rda_path)
    
    # In yan.rda, there are typically two objects: 'yan' and 'ann'
    if 'yan' not in result or 'ann' not in result:
        raise ValueError(f"The .rda file does not contain expected 'yan' and 'ann' objects. Found: {result.keys()}")
        
    # 2. Extract Expression Matrix (yan)
    # R: Genes x Cells
    # Python: DataFrame where Index=Genes, Columns=Cells
    yan_df = result['yan']
    
    # Preprocessing to match R's SingleCellExperiment construction:
    # R: logcounts = log2(as.matrix(yan) + 1)
    # Note: 'yan' in R is usually raw counts or RPKM. 
    # For SC3-Nature methods-2017, we need log-transformed data.
    X_raw = yan_df.values.T # Transpose to (Cells, Genes)
    X = np.log2(X_raw + 1)
    
    gene_names = yan_df.index.to_numpy()
    cell_names = yan_df.columns.to_numpy()
    
    # 3. Extract Annotations (ann)
    # ann is a dataframe with cell types
    ann_df = result['ann']
    
    # The 'ann' object usually has a column like 'cell_type1'
    # We map these strings to integers for Y
    if not ann_df.empty:
        # Assuming the first column is the label
        label_col = ann_df.columns[0]
        labels_raw = ann_df[label_col].values
        
        # Convert string labels to integers
        unique_labels = np.unique(labels_raw)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        Y = np.array([label_map[l] for l in labels_raw])
    else:
        Y = None
        
    return X, Y, gene_names, cell_names
