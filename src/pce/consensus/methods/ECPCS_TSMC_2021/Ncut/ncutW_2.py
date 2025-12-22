import numpy as np
from .ncut_2 import ncut_2
from .discretisation import discretisation

def ncutW_2(W, nbcluster):
    """
    Computes Ncut eigenvectors and discretizes them.
    
    Corresponds to ncutW_2.m
    
    Parameters:
    -----------
    W : numpy.ndarray
        Similarity matrix.
    nbcluster : int
        Number of clusters.
        
    Returns:
    --------
    NcutDiscrete : numpy.ndarray
        Discretized Ncut vectors (Dense).
    NcutEigenvectors : numpy.ndarray
        Normalized/Rotated Ncut eigenvectors.
    NcutEigenvalues : numpy.ndarray
        Ncut eigenvalues.
    """
    
    # compute continuous Ncut eigenvectors
    # [NcutEigenvectors,NcutEigenvalues] = ncut_2(W,nbcluster);
    NcutEigenvectors, NcutEigenvalues = ncut_2(W, nbcluster)
    
    # compute discretize Ncut vectors
    # [NcutDiscrete,NcutEigenvectors] =discretisation(NcutEigenvectors);
    NcutDiscrete, NcutEigenvectors = discretisation(NcutEigenvectors)
    
    # NcutDiscrete = full(NcutDiscrete);
    if hasattr(NcutDiscrete, 'toarray'):
        NcutDiscrete = NcutDiscrete.toarray()
        
    return NcutDiscrete
