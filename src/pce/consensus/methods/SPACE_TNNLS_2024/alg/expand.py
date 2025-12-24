import numpy as np
from .transclos import transclos
from .constructEdge import construct_edge

def expand(cons, V_old=None, cannot=None):
    """
    Expand constraints.
    
    Parameters:
    cons : numpy.ndarray
        Constraints array [i, j, label]. Label 1 for ML, -1 for CL.
    V_old : list, optional
        Previous mustlink sets.
    cannot : numpy.ndarray, optional
        Previous cannotlink constraints.
        
    Returns:
    V : list
        Updated mustlink sets.
    E : numpy.ndarray
        Edge matrix.
    cannotlink : numpy.ndarray
        Updated cannotlink constraints.
    """
    num_cons = cons.shape[0]
    label = cons[:, 2]
    
    # MATLAB: idx=find(label==1);
    idx = np.where(label == 1)[0]
    
    # MATLAB: idx_cannot=setdiff(1:num_cons,idx);
    # In Python 0-based:
    all_indices = np.arange(num_cons)
    idx_cannot = np.setdiff1d(all_indices, idx)
    
    num_must = len(idx)
    mustlink = []
    
    # MATLAB: mustlink{i}=cons(idx(i),1:2);
    for i in range(num_must):
        # Store as set for transclos
        mustlink.append(set(cons[idx[i], :2].astype(int)))
        
    if len(idx_cannot) > 0:
        cannotlink = cons[idx_cannot, :2].astype(int)
    else:
        cannotlink = np.empty((0, 2), dtype=int)
        
    if V_old is None:
        if len(idx) == 0:
            mustlink = []
        else:
            mustlink = transclos(mustlink)
        
        E, V, cannotlink = construct_edge(mustlink, cannotlink)
        
    else:
        # cannotlink=[cannot;cannotlink];
        if cannot is not None and len(cannot) > 0:
            if len(cannotlink) > 0:
                cannotlink = np.vstack([cannot, cannotlink])
            else:
                cannotlink = cannot
                
        if len(idx) == 0:
            # mustlink=[]; -> handled by empty list init
            # MATLAB logic: mustlink=[V_old, mustlink] -> [V_old]
            mustlink = V_old
        else:
            # mustlink=[V_old,mustlink];
            mustlink = V_old + mustlink
            
        mustlink = transclos(mustlink)
        E, V, cannotlink = construct_edge(mustlink, cannotlink)
        
    return V, E, cannotlink
