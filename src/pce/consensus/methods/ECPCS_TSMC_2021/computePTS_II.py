import numpy as np

def computePTS_II(S, paraT):
    """
    Performs random walk on S and obtains new similarity matrix.
    
    Corresponds to computePTS_II.m
    
    Parameters:
    -----------
    S : numpy.ndarray
        Similarity matrix (N, N).
    paraT : int
        Number of steps of random walk.
        
    Returns:
    --------
    newS : numpy.ndarray
        The newly obtained similarity matrix.
    """
    S = S.copy()
    N = S.shape[0]
    
    # for i = 1:size(S,1), S(i,i)=0; end
    np.fill_diagonal(S, 0)
    
    # rowSum = sum(S,2);
    rowSum = np.sum(S, axis=1)
    
    # rowSums(rowSums==0)=-1;
    rowSums = rowSum.copy()
    rowSums[rowSums == 0] = -1.0
    
    # P = S./rowSums;
    # Python broadcasting handles the repmat
    P = S / rowSums[:, None]
    
    # P(P<0)=0;
    P[P < 0] = 0
    
    # Compute PTS
    tmpP = P.copy()
    inProdP = np.dot(P, P.T)
    
    for ii in range(paraT - 1):
        tmpP = np.dot(tmpP, P)
        inProdP = inProdP + np.dot(tmpP, tmpP.T)
        
    # newS = inProdP./sqrt(inProdPii.*inProdPjj);
    inProdPii = np.diag(inProdP)
    
    # Outer product for denominator sqrt(d_i * d_j)
    denom_sq = np.outer(inProdPii, inProdPii)
    denom = np.sqrt(denom_sq)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        newS = inProdP / denom
        
    newS[np.isnan(newS)] = 0.0
    
    # sr = sum(P'); -> Matlab sums columns (which corresponds to row sums of P because P' is transpose)
    # Wait, Matlab sum(A) sums columns. sum(P') sums columns of P'.
    # Columns of P' are rows of P.
    # So sr is row sums of P.
    sr = np.sum(P, axis=1)
    
    isolatedIdx = np.where(sr < 1e-10)[0]
    if len(isolatedIdx) > 0:
        newS[isolatedIdx, :] = 0
        newS[:, isolatedIdx] = 0
        
    return newS
