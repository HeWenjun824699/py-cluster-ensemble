import numpy as np


def pur_fun(Y, predY):
    """
    Calculate purity.
    """
    Y = np.array(Y)
    predY = np.array(predY)
    
    predLidx = np.unique(predY)
    pred_classnum = len(predLidx)
    
    correnum = 0
    
    for ci in range(pred_classnum):
        incluster = Y[predY == predLidx[ci]]
        
        if len(incluster) == 0:
            continue
            
        # inclunub = hist(incluster, 1:max(incluster)) in MATLAB
        # Essentially finding the count of the most frequent class in this cluster
        
        # np.unique returns counts with return_counts=True
        values, counts = np.unique(incluster, return_counts=True)
        
        if len(counts) == 0:
            inclunub = 0
        else:
            inclunub = np.max(counts)
            
        correnum += inclunub
        
    Purity = correnum / len(predY)
    return Purity
