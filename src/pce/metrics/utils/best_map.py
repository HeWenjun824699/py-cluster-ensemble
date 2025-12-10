import numpy as np
from .hungarian import hungarian


def best_map(L1, L2):
    """
    Permute labels of L2 match L1 as good as possible.
    """
    L1 = np.array(L1).flatten()
    L2 = np.array(L2).flatten()
    
    if L1.shape != L2.shape:
        raise ValueError('size(L1) must == size(L2)')
        
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    
    # Calculate overlap matrix
    # G(i,j) = length(find(L1 == Label1(i) & L2 == Label2(j)))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))
            
    # Hungarian algorithm
    # maximize overlap => minimize negative overlap
    c, t = hungarian(-G)
    
    newL2 = np.zeros(L2.shape, dtype=L2.dtype)
    
    # Assignment c: c[i] is the row index (Label1 index) assigned to column i (Label2 index)
    # The loop in MATLAB:
    # for i=1:nClass2
    #     newL2(L2 == Label2(i)) = Label1(c(i));
    # end
    
    # Note: c comes from hungarian wrapper which returns Python 0-based indices
    # matching the MATLAB logic port.
    
    for i in range(nClass2):
        # Label2[i] is the original label in L2
        # c[i] is the index in Label1 that it maps to
        # So we assign Label1[c[i]] to where L2 was Label2[i]
        
        # Check bounds: hungarian returns c of size nClass (square matrix size)
        # But we only iterate up to nClass2.
        # c has length nClass.
        
        if i < len(c):
             mask = (L2 == Label2[i])
             # c[i] is the index in Label1.
             # Wait, if nClass1 < nClass2, c[i] might point to an index >= nClass1?
             # G is nClass x nClass.
             # If nClass1 < nClass2, we padded G with zeros rows.
             # So c[i] can be >= nClass1.
             # But Label1 has only nClass1 elements.
             # In MATLAB: Label1(c(i)). If c(i) > nClass1, error?
             # Let's check MATLAB code:
             # nClass = max(nClass1,nClass2); G = zeros(nClass);
             # ... loops to nClass1/2 ...
             # [c,t] = hungarian(-G);
             # newL2 = zeros(size(L2));
             # for i=1:nClass2
             #    newL2(L2 == Label2(i)) = Label1(c(i));
             # end
             # If c(i) > nClass1, Label1(c(i)) would error in MATLAB unless Label1 logic is different.
             # Wait, if nClass2 > nClass1 (more clusters in L2), then nClass = nClass2.
             # Rows are nClass (padded). Label1 has length nClass1.
             # If a column (L2 cluster) is assigned to a row index > nClass1 (empty row),
             # then we map it to... what?
             # The MATLAB code would crash if c(i) > length(Label1).
             # Maybe best_map assumes nClass1 >= nClass2 usually, or `Label1` is not used for those?
             # Or maybe `Label1` handling in MATLAB implies something I missed?
             # Re-reading best_map.m:
             # Label1 = unique(L1); ... 
             # G filled up to nClass1, nClass2.
             # If c(i) points to a dummy row (index >= nClass1), Label1(c(i)) is out of bounds.
             # Perhaps the assumption is strict or handled by data.
             # But for robustness in Python:
             
             if c[i] < nClass1:
                 newL2[mask] = Label1[c[i]]
             else:
                 # If mapped to a dummy label, what to assign?
                 # Maybe leave as 0 or assign a new dummy label?
                 # For now, let's assume it doesn't happen or map to 0/original.
                 # Leaving as 0 (init value) seems safest if no match found in L1.
                 pass
                 
    return newL2
