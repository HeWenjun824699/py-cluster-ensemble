import numpy as np

def getECI(bcs, baseClsSegs, para_theta):
    """
    Corresponds to getECI.m
    """
    M = bcs.shape[1]
    ETs = getAllClsEntropy(bcs, baseClsSegs)
    
    # ECI = exp(-ETs./para_theta./M);
    ECI = np.exp(-ETs / para_theta / M)
    return ECI

def getAllClsEntropy(bcs, baseClsSegs):
    # baseClsSegs = baseClsSegs';
    # In Matlab, input baseClsSegs is (nCls, N). Transposed becomes (N, nCls).
    # We will just transpose it for processing to match the logic.
    baseClsSegs_T = baseClsSegs.T
    
    # [~, nCls] = size(baseClsSegs); (after transpose)
    nCls = baseClsSegs_T.shape[1]
    
    Es = np.zeros(nCls)
    for i in range(nCls):
        # partBcs = bcs(baseClsSegs(:,i)~=0,:);
        # baseClsSegs_T[:, i] corresponds to the i-th cluster indicator column
        mask = baseClsSegs_T[:, i] != 0
        partBcs = bcs[mask, :]
        
        Es[i] = getOneClsEntropy(partBcs)
    
    return Es

def getOneClsEntropy(partBcs):
    # E = 0;
    E = 0.0
    # for i = 1:size(partBcs,2)
    for i in range(partBcs.shape[1]):
        tmp = partBcs[:, i]
        # uTmp = unique(tmp);
        uTmp, counts = np.unique(tmp, return_counts=True)
        
        if len(uTmp) <= 1:
            continue
        
        # cnts = zeros(size(uTmp)); ... cnts = cnts./sum(cnts(:));
        # np.unique with return_counts gives us the counts directly
        probs = counts / np.sum(counts)
        
        # E = E-sum(cnts.*log2(cnts));
        E = E - np.sum(probs * np.log2(probs))
        
    return E
