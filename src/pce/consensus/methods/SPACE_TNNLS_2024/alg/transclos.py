def transclos(mustlink):
    """
    Transitive closure of must-link constraints.
    
    Parameters:
    mustlink : list of sets or lists
        Each element is a set/list of connected indices.
        
    Returns:
    mustlink : list of sets
        Merged sets representing transitive closure.
    """
    # MATLAB uses cells of arrays. Python list of sets is appropriate.
    # Convert input to list of sets if they are not already
    mustlink = [set(m) for m in mustlink]
    
    start = 0
    num_must = len(mustlink)
    
    while start < num_must:
        flag = 0
        i = start + 1
        
        # We need to be careful about modifying list while iterating
        # Use while loop similar to MATLAB
        while i < num_must:
            # MATLAB: intersect(mustlink{start}, mustlink{i})
            if not mustlink[start].isdisjoint(mustlink[i]):
                # MATLAB: union(mustlink{start}, mustlink{i})
                mustlink[start] = mustlink[start].union(mustlink[i])
                # MATLAB: mustlink(i)=[];
                mustlink.pop(i)
                num_must -= 1
                flag = 1
            else:
                i += 1
                
        if flag == 0:
            start += 1
            
    return mustlink
