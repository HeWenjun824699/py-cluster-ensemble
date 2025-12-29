from .core_methods import SC3
from .core_functions import estkTW

__all__ = ["SC3", "estkTW", "sc3"]

def sc3(data, n_clusters=None, biology=False, **kwargs):
    """
    Functional interface for SC3_R.
    
    Parameters
    ----------
    data : np.ndarray
        (n_cells, n_genes) Input matrix.
    n_clusters : int, optional
        Number of clusters. If None, estimated automatically.
    biology : bool, default=False
        Whether to calculate biological features.
    **kwargs : 
        Additional arguments for SC3_R class (d_region_min, seed, svm_max, etc.)
        
    Returns
    -------
    np.ndarray
        Cluster labels.
    """
    model = SC3(data, **kwargs)
    return model.run(n_clusters=n_clusters, biology=biology)
