import numpy as np
from scipy import sparse
from .EuDist2 import EuDist2
from .NormalizeFea import NormalizeFea

def constructW(fea, options=None):
    """
    Constructs the affinity graph W.
    """
    if options is None:
        options = {}
        
    # Defaults
    if 'bNormalized' not in options:
        options['bNormalized'] = 0
    if 'NeighborMode' not in options:
        options['NeighborMode'] = 'KNN'
    if 'WeightMode' not in options:
        options['WeightMode'] = 'HeatKernel'
    if 'bSelfConnected' not in options:
        options['bSelfConnected'] = 0
        
    # KNN defaults
    if options['NeighborMode'].lower() == 'knn':
        if 'k' not in options:
            options['k'] = 5
            
    bBinary = 0
    bCosine = 0
    if options['WeightMode'].lower() == 'binary':
        bBinary = 1
    elif options['WeightMode'].lower() == 'cosine':
        bCosine = 1
    elif options['WeightMode'].lower() == 'heatkernel':
        if 't' not in options:
            nSmp = fea.shape[0]
            if nSmp > 3000:
                # Random sample
                indices = np.random.choice(nSmp, 3000, replace=False)
                D = EuDist2(fea[indices, :])
            else:
                D = EuDist2(fea)
            options['t'] = np.mean(D)
            
    if 'gnd' in options:
        nSmp = len(options['gnd'])
    else:
        nSmp = fea.shape[0]
        
    # Supervised Mode logic skipped for brevity unless needed (CEAM uses KNN)
    # If needed, I would add it here.
    
    if bCosine and not options['bNormalized']:
        Normfea = NormalizeFea(fea)
    else:
        Normfea = fea
        
    # KNN Mode
    if options['NeighborMode'].lower() == 'knn' and options['k'] > 0:
        k = options['k']
        
        # We need k+1 neighbors because the point itself is included usually in search
        # The MATLAB code loops over blocks. We'll do a simpler full matrix approach 
        # unless memory is an issue, but to match "Strict consistency" with the *result*,
        # we just need to produce the same W.
        
        # Distance/Similarity calculation
        if bCosine:
            # Cosine similarity
            # dist = Normfea * Normfea'
            if sparse.issparse(Normfea):
                dist = Normfea @ Normfea.T
                dist = dist.toarray() # KNN usually requires dense to sort
            else:
                dist = Normfea @ Normfea.T
        else:
            # Euclidean
            dist = EuDist2(fea, bSqrt=False) # squared distance for sorting is fine/faster
            
        # Find Neighbors
        # MATLAB: [dump, idx] = sort(dist, 2) or min/max
        
        if bCosine:
            # Higher is better. Sort descending.
            # We want top k+1
            # np.argsort sorts ascending. 
            idx = np.argsort(-dist, axis=1)[:, :k+1]
            dump = -np.take_along_axis(-dist, idx, axis=1) # Get values
        else:
            # Euclidean: Lower is better. Sort ascending.
            idx = np.argsort(dist, axis=1)[:, :k+1]
            dump = np.take_along_axis(dist, idx, axis=1)
            
        # Construct G (triplets)
        # G(:, 1) = repmat(smpIdx, ...)
        # G(:, 2) = idx
        # G(:, 3) = values
        
        n_entries = nSmp * (k + 1)
        row_inds = np.repeat(np.arange(nSmp), k + 1)
        col_inds = idx.flatten()
        
        if not bBinary:
            if bCosine:
                vals = dump.flatten()
            else:
                # HeatKernel
                vals = np.exp(-dump.flatten() / (2 * options['t']**2))
        else:
            vals = np.ones(n_entries)
            
        W = sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(nSmp, nSmp))
        W = W.tolil() # Convert to LIL for easy modification if needed or max
        
        # Handling bSemiSupervised skipped
        
        if not options['bSelfConnected']:
            W.setdiag(0)
            
        # Symmetrize
        # MATLAB: W = max(W, W')
        # Scipy sparse max is tricky. Convert to CSR/CSC.
        W = W.tocsr()
        W = W.maximum(W.T)
        
        return W

    # Complete Graph Mode (k=0)
    if options['NeighborMode'].lower() == 'knn' and options['k'] == 0:
        if bBinary:
             raise ValueError('Binary weight can not be used for complete graph!')
        elif options['WeightMode'].lower() == 'heatkernel':
            W = EuDist2(fea, bSqrt=False)
            W = np.exp(-W / (2 * options['t']**2))
        elif bCosine:
            W = Normfea @ Normfea.T
            if sparse.issparse(W):
                W = W.toarray()
        
        if not options['bSelfConnected']:
            np.fill_diagonal(W, 0)
            
        W = np.maximum(W, W.T)
        return sparse.csr_matrix(W)

    return None
