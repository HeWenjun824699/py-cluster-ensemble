import numpy as np

def construct_edge(mustlink, cannotlink):
    """
    Construct edge matrix E and update mustlink/cannotlink constraints.
    
    Parameters:
    mustlink : list of sets/lists
    cannotlink : list of lists or numpy.ndarray (N x 2)
    
    Returns:
    E : numpy.ndarray
        Edge matrix between clusters.
    mustlink : list of sets
    cannotlink : numpy.ndarray
        Updated cannotlink constraints.
    """
    # Ensure mustlink is list of sets (from transclos or input)
    mustlink = [set(m) for m in mustlink]
    
    len_must = len(mustlink)
    # E=zeros(len_must,len_must);
    E = np.zeros((len_must, len_must))
    
    num_cannot = len(cannotlink)
    remove = []
    
    # Iterate through cannotlink constraints
    for i in range(num_cannot):
        ii = cannotlink[i][0]
        jj = cannotlink[i][1]
        
        ii_idx = -1 # Using -1 for 0-based indexing equivalent of 0 check
        jj_idx = -1
        
        for j in range(len_must):
            if ii in mustlink[j]:
                ii_idx = j
            if jj in mustlink[j]:
                jj_idx = j
                
            if ii_idx > -1 and jj_idx > -1:
                # If E(ii_idx, jj_idx)==1
                if E[ii_idx, jj_idx] == 1:
                    remove.append(i)
                else:
                    E[ii_idx, jj_idx] = 1
                    E[jj_idx, ii_idx] = 1
                break
        
        # Logic for adding new clusters if not found
        # Note: MATLAB array growth is dynamic. Python lists are dynamic, 
        # but E is numpy array, needs resize/padding.
        
        if ii_idx == -1 and jj_idx > -1:
            # mustlink{len_must+1}=[ii];
            mustlink.append({ii})
            
            # E=[E,zeros(len_must,1);zeros(1,len_must),0];
            # E(len_must+1,jj_idx)=1; ...
            new_E = np.zeros((len_must + 1, len_must + 1))
            new_E[:len_must, :len_must] = E
            E = new_E
            
            E[len_must, jj_idx] = 1
            E[jj_idx, len_must] = 1
            len_must += 1
            
        if ii_idx > -1 and jj_idx == -1:
            mustlink.append({jj})
            
            new_E = np.zeros((len_must + 1, len_must + 1))
            new_E[:len_must, :len_must] = E
            E = new_E
            
            E[len_must, ii_idx] = 1
            E[ii_idx, len_must] = 1
            len_must += 1
            
        if ii_idx == -1 and jj_idx == -1:
            mustlink.append({ii})
            mustlink.append({jj})
            
            # E=[E,zeros(len_must,2);zeros(2,len_must),[0,1;1,0]];
            new_E = np.zeros((len_must + 2, len_must + 2))
            new_E[:len_must, :len_must] = E
            new_E[len_must, len_must+1] = 1
            new_E[len_must+1, len_must] = 1
            E = new_E
            
            len_must += 2
            
    # cannotlink(remove,:)=[];
    if remove:
        cannotlink = np.delete(cannotlink, remove, axis=0)
        
    return E, mustlink, cannotlink
