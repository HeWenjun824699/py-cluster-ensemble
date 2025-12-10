import numpy as np


def mutual_info(L1, L2):
    """
    Calculate normalized mutual information.
    MIhat = MI / max(H1, H2)
    """
    L1 = np.array(L1).flatten()
    L2 = np.array(L2).flatten()
    
    if L1.shape != L2.shape:
        raise ValueError('size(L1) must == size(L2)')
    
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    
    # Smoothing logic from MATLAB code
    if nClass2 < nClass1:
        L1 = np.concatenate([L1, Label1])
        L2 = np.concatenate([L2, Label1]) # Adding Label1 to L2? MATLAB code: L2 = [L2; Label]; Label is from L1.
    elif nClass2 > nClass1:
        L1 = np.concatenate([L1, Label2])
        L2 = np.concatenate([L2, Label2])
        
    Label = np.unique(L1)
    nClass = len(Label)
    
    # Map labels to 0..nClass-1
    # We can use np.searchsorted or just re-unique
    # To match MATLAB's strict G matrix construction:
    # G(i,j) = sum(L1 == Label(i) & L2 == Label(j))
    # We need to ensure we use the same Label set for both after smoothing
    
    # Optimization: Use indices
    u1, inv1 = np.unique(L1, return_inverse=True) # u1 should be same as Label if nClass1>=nClass2 originally
    u2, inv2 = np.unique(L2, return_inverse=True)
    
    # The MATLAB code assumes 'Label' (from L1) is the unified label set if nClass2 < nClass1
    # and Label2 is used if nClass2 > nClass1.
    # Basically it ensures square matrix by adding missing classes.
    # Let's stick to the logic:
    
    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == Label[i]) & (L2 == Label[j]))
            
    sumG = np.sum(G)
    
    P1 = np.sum(G, axis=1) / sumG
    P2 = np.sum(G, axis=0) / sumG
    
    if np.any(P1 == 0) or np.any(P2 == 0):
        # This shouldn't happen with the smoothing logic unless empty
        raise ValueError('Smooth fail!')
    
    H1 = np.sum(-P1 * np.log2(P1))
    H2 = np.sum(-P2 * np.log2(P2))
    
    P12 = G / sumG
    
    # PPP = P12 / (P2 * P1)
    # Using broadcasting
    # P2 is row, P1 is col.
    # P1: (n, ), P2: (n, )
    # P1[:, None] * P2[None, :] -> (n, n) matrix of P1(i)*P2(j)
    denominator = P1[:, None] * P2[None, :]
    PPP = P12 / denominator
    
    # Avoid log(0) or division issues
    PPP[np.abs(PPP) < 1e-12] = 1
    
    MI = np.sum(P12 * np.log2(PPP))
    
    MIhat = MI / max(H1, H2)
    
    return float(MIhat)
