import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


def estkTW(dataset):
    """
    Estimate optimal k using Tracy-Widom theory.
    
    R equivalent: estkTW in CoreFunctions.R
    
    Parameters
    ----------
    dataset : np.ndarray
        (n_cells, n_genes) - CAUTION: R code expects (genes, cells).
        We assume input here is (n_cells, n_genes) to match Python convention.
        
    Returns
    -------
    int
        Estimated k.
    """
    # R: p <- ncol(dataset) (cells), n <- nrow(dataset) (genes)
    # Python: dataset is (cells, genes).
    p = dataset.shape[0] # cells
    n = dataset.shape[1] # genes
    
    # R: x <- scale(dataset)
    # R scale works on columns. Dataset is (genes, cells).
    # So it scales each cell (column).
    # Python: We need to scale each row (cell).
    # StandardScaler scales columns.
    scaler = StandardScaler()
    # Fit on dataset.T (genes, cells) to scale cells? 
    # Or just use axis.
    # We want: for each cell (row), mean=0, std=1.
    x_scaled = scaler.fit_transform(dataset.T).T # (cells, genes)
    
    # muTW <- (sqrt(n - 1) + sqrt(p))^2
    muTW = (np.sqrt(n - 1) + np.sqrt(p))**2
    
    # sigmaTW <- (sqrt(n - 1) + sqrt(p)) * (1/sqrt(n - 1) + 1/sqrt(p))^(1/3)
    sigmaTW = (np.sqrt(n - 1) + np.sqrt(p)) * (1/np.sqrt(n - 1) + 1/np.sqrt(p))**(1/3)
    
    # sigmaHatNaive <- tmult(x)
    # R tmult(x) = x %*% t(x) (since x is genes x cells, result is genes x genes? Wait.)
    # Let's check cppFunctions.cpp: returns x.t() * x.
    # If x is (genes, cells), x.t() is (cells, genes). x.t() * x is (cells, cells).
    # Correct.
    # Python x_scaled is (cells, genes).
    # We want (cells, cells).
    # So x_scaled * x_scaled.T
    
    sigmaHatNaive = x_scaled @ x_scaled.T
    
    # bd <- 3.273 * sigmaTW + muTW
    bd = 3.273 * sigmaTW + muTW
    
    # evals <- eigen(sigmaHatNaive, symmetric = TRUE, only.values = TRUE)$values
    # R eigen returns sorted decreasing.
    evals = eigh(sigmaHatNaive, eigvals_only=True)
    # eigh returns ascending.
    
    # k <- 0; for(i in 1:length(evals)){ if(evals[i] > bd) k++ }
    k = np.sum(evals > bd)
    
    return k
