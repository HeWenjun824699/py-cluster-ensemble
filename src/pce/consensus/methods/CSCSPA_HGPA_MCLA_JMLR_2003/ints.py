import numpy as np

def ints(s, tafter=100000000):
    tbefore = np.sum(s)
    if tbefore != 0:
        s = np.round(s * (tafter / tbefore))
    return s
