import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs
from sklearn.cluster import KMeans
import warnings

def performSC(W, clsNum):
    """
    Performs Spectral Clustering.
    
    Args:
        W: Affinity/Similarity matrix (N x N)
        clsNum: Number of clusters (int or list/array of ints)
        
    Returns:
        group: Cluster labels (if clsNum is int) or list of labels (if clsNum is list)
        eigengap: Eigenvalue gaps
    """
    warnings.filterwarnings("ignore")
    
    N = W.shape[0]
    
    # Calculate degree matrix
    # degs = sum(W, 2)
    # Ensure W is handled correctly whether dense or sparse
    if sp.issparse(W):
        degs = np.array(W.sum(axis=1)).flatten()
    else:
        degs = np.sum(W, axis=1)
    
    # D = sparse(1:size(W, 1), 1:size(W, 2), degs)
    D = sp.diags(degs, 0, shape=(N, N), format='csc')
    
    # Compute unnormalized Laplacian
    # L = D - W
    if sp.issparse(W):
        L = D - W
    else:
        L = D - sp.csc_matrix(W)
        
    if np.isscalar(clsNum):
        clsNum_list = [clsNum]
    else:
        clsNum_list = clsNum
        
    k = np.max(clsNum_list)
    
    # Avoid dividing by zero
    # degs(degs == 0) = eps
    eps = np.finfo(float).eps
    degs[degs == 0] = eps
    
    # Calculate D^(-1/2)
    # D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2))
    inv_sqrt_degs = 1.0 / (degs ** 0.5)
    D_inv_sqrt = sp.diags(inv_sqrt_degs, 0, shape=(N, N), format='csc')
    
    # Calculate normalized Laplacian
    # L = D * L * D  (here D is D^(-1/2))
    L = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Compute the eigenvectors corresponding to the k smallest eigenvalues
    # [U, eigenvalue] = eigs(L, k, eps); 
    # Use 'SA' (Smallest Algebraic) for symmetric L.
    try:
        eigenvalues, U = eigsh(L, k=k, which='SA')
    except:
        # Fallback
        eigenvalues, U = eigs(L, k=k, which='SR')
        eigenvalues = np.real(eigenvalues)
        U = np.real(U)

    # Sort eigenvalues/vectors
    # [a,b] = sort(diag(eigenvalue),'ascend');
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    U = U[:, sorted_indices]
    
    # eigengap = abs(diff(diag(eigenvalue)))
    eigengap = np.abs(np.diff(eigenvalues))
    
    # U = U(:,1:k)
    U = U[:, :k]
    
    Cluster = []
    
    # To map back to input order if clsNum is a list, we need to be careful.
    # The Matlab code uses `Cluster{Cindex} = temp` where Cindex is finding indices.
    # It constructs a cell array.
    
    # We will use a dictionary to store results for each k temporarily
    temp_results = {}
    
    for ck in clsNum_list:
        UU = U[:, :ck]
        
        # UU = UU./repmat(sqrt(sum(UU.^2,2)),1,size(UU,2));
        row_sums = np.sqrt(np.sum(UU**2, axis=1))
        row_sums[row_sums == 0] = eps
        UU = UU / row_sums[:, np.newaxis]
        
        # temp = kmeans(UU,ck,'MaxIter',100,'Replicates',3);
        kmeans = KMeans(n_clusters=ck, max_iter=100, n_init=3)
        labels = kmeans.fit_predict(UU)
        
        # Matlab returns 1-based labels, Python 0-based.
        # Adding 1 for consistency with Matlab.
        temp_results[ck] = labels + 1
        
    if np.isscalar(clsNum):
        group = temp_results[clsNum]
    else:
        # Return list in order of clsNum_list
        group = [temp_results[k] for k in clsNum_list]
        
    # return group, eigengap
    return group
