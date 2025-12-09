import numpy as np

def checks(s):
    if s is not None and s.size > 0:
        # Check for complex values
        if np.iscomplexobj(s):
            print('checks: complex similarities found')
            s = s.real
            print('checks: using real component')
        
        # Check for square matrix
        if s.shape[0] != s.shape[1]:
            print('checks: s is not square')
            min_dim = min(s.shape)
            s = s[:min_dim, :min_dim]
            print('checks: using indrawn square')
        
        mas = np.max(s)
        mis = np.min(s)
        
        if mas > 1 or mis < 0:
            print(f'checks: similarity more than 1 or less than 0 detected: values {mis} to {mas}')
            s[s < 0] = 0
            s[s > 1] = 1
            print('checks: bounded')
            
        if np.any(np.isinf(s) | np.isnan(s)):
            print('checks: non-finite similarity detected !!! (serious)')
            s[np.isinf(s) | np.isnan(s)] = 0
            print('checks: made zero !!! (serious)')

        # Symmetry check
        if np.any(s != s.T):
            print('checks: s is not symmetric')
            s = (s + s.T) / 2
            print('checks: symmetrised')
            
        # Self-similarity check
        if s.shape[0] == s.shape[1]:
            if np.any(np.diag(s) != 1):
                print('checks: self-similarity s(i,i) not always 1')
                np.fill_diagonal(s, 1)
                print('checks: diagonal made 1')
    else:
        print('checks: empty similarity matrix')
        
    return s
