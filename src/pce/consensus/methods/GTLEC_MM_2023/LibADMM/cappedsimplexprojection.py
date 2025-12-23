import numpy as np

def cappedsimplexprojection(y0, k):
    """
    min 0.5||x-y0||, s.t. 0<=x<=1, sum x_i = k;
    """
    n = len(y0)
    x = np.zeros(n)

    if k < 0 or k > n:
        raise ValueError('the sum constraint is infeasible!')

    if k == 0:
        e = 0.5 * np.sum((x - y0)**2)
        return x #, e

    if k == n:
        x = np.ones(n)
        e = 0.5 * np.sum((x - y0)**2)
        return x #, e

    # Sort y0
    idx = np.argsort(y0) # Default is ascending
    y = y0[idx]

    # Test the possibility of a==b are integers.
    # Check if k is integer (close enough)
    if np.isclose(k, np.round(k)):
        b_int = int(n - k) # MATLAB: b=n-k. Python index 0-based.
        # MATLAB: y(b+1)-y(b)>=1
        # Python: y[b] - y[b-1] >= 1?
        # Let's align indices.
        # MATLAB y is 1..n. b is index. y(b) is b-th smallest.
        # Python y is 0..n-1. y[b-1] is b-th smallest.
        # But b = n - k.
        # Example n=5, k=2. b=3.
        # MATLAB: Check y(4) - y(3) >= 1. Set x(idx(4:end)) = 1. i.e. x(idx(b+1:end)) = 1.
        # Python: Check y[b] - y[b-1] >= 1. Set x[idx[b:]] = 1.
        
        # Be careful with indices.
        # If b=0 (k=n), handled above.
        # If b=n (k=0), handled above.
        
        if 0 < b_int < n:
             if y[b_int] - y[b_int - 1] >= 1:
                 x[idx[b_int:]] = 1
                 # e = 0.5 * np.sum((x - y0)**2)
                 return x #, e

    # Assume a=0.
    s = np.cumsum(y)
    # y = [y; inf] in MATLAB.
    y_inf = np.concatenate([y, [np.inf]])
    
    # MATLAB loop b=1:n
    for b in range(1, n + 1):
        # b is 1-based count.
        # s(b) is sum of first b elements. Python s[b-1].
        sb = s[b-1]
        
        # gamma = (k + b - n - s(b)) / b
        gamma = (k + b - n - sb) / b
        
        # Check conditions
        # MATLAB: ((y(1)+gamma)>0) && ((y(b)+gamma)<1) && ((y(b+1)+gamma)>=1)
        # Python: y[0] ... y[b-1] ... y[b] (which is y_inf[b])
        
        c1 = (y_inf[0] + gamma) > 0
        c2 = (y_inf[b-1] + gamma) < 1
        c3 = (y_inf[b] + gamma) >= 1
        
        if c1 and c2 and c3:
            # xtmp = [y(1:b)+gamma; ones(n-b,1)]
            xtmp = np.concatenate([y_inf[:b] + gamma, np.ones(n - b)])
            x[idx] = xtmp
            # e = 0.5 * np.sum((x - y0)**2)
            return x #, e

    # Now a >= 1
    # MATLAB: for a=1:n, for b=a+1:n
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            # gamma = (k + b - n + s(a) - s(b)) / (b - a)
            # s(a) -> s[a-1], s(b) -> s[b-1]
            sa = s[a-1]
            sb = s[b-1]
            
            gamma = (k + b - n + sa - sb) / (b - a)
            
            # Conditions
            # MATLAB: ((y(a)+gamma)<=0) && ((y(a+1)+gamma)>0) && ((y(b)+gamma)<1) && ((y(b+1)+gamma)>=1)
            # Python indices:
            # y(a) -> y_inf[a-1]
            # y(a+1) -> y_inf[a]
            # y(b) -> y_inf[b-1]
            # y(b+1) -> y_inf[b]
            
            c1 = (y_inf[a-1] + gamma) <= 0
            c2 = (y_inf[a] + gamma) > 0
            c3 = (y_inf[b-1] + gamma) < 1
            c4 = (y_inf[b] + gamma) >= 1
            
            if c1 and c2 and c3 and c4:
                # xtmp = [zeros(a,1); y(a+1:b)+gamma; ones(n-b,1)]
                # y(a+1:b) -> y_inf[a : b] (since python slice is start:end_exclusive)
                
                part1 = np.zeros(a)
                part2 = y_inf[a:b] + gamma
                part3 = np.ones(n - b)
                
                xtmp = np.concatenate([part1, part2, part3])
                x[idx] = xtmp
                # e = 0.5 * np.sum((x - y0)**2)
                return x #, e

    return x # Should not be reached if feasible
