import numpy as np
from .utils.best_map import best_map
from .utils.mutual_info import mutual_info
from .utils.pur_fun import pur_fun
from .utils.rand_index import rand_index
from .utils.compute_f import compute_f
from .utils.balance_evl import balance_evl


def evaluation(y, Y):
    """
    Evaluates clustering performance.
    
    Parameters:
    y : Ground truth labels
    Y : Predicted labels
    
    Returns:
    res : list or array
          [acc, nmi, purity, AR, RI, MI, HI, fscore, precision, recall, entropy, SDCS, RME, bal]
    """
    y = np.array(y).flatten()
    Y = np.array(Y).flatten()
    
    # Map Y to match y
    newIndx = best_map(Y, y) # Note: MATLAB call was best_map(Y, y). best_map(L1, L2). L1=Y (ref?), L2=y (target to permute?). 
    # MATLAB: function [newL2] = best_map(L1,L2). 
    # Returns permuted L2.
    # my_eval_y: [newIndx] = best_map(Y,y);
    # So it permutes 'y' to match 'Y'?
    # acc = mean(Y==newIndx);
    # If newIndx is permuted 'y', then we compare Y (pred) with permuted y (truth).
    # Yes.
    
    acc = np.mean(Y == newIndx)
    nmi = mutual_info(Y, newIndx)
    purity = pur_fun(Y, newIndx)
    AR, RI, MI, HI = rand_index(Y, newIndx)
    fscore, precision, recall = compute_f(Y, newIndx)
    
    # Balance Eval
    # nCluster = length(unique(Y));
    # MATLAB: FF = zeros(nSmp, nCluster); for iSmp=1:nSmp, FF(iSmp, y(iSmp))=1; end
    # Note: MATLAB uses 'y' (the second arg, original truth?) for BalanceEvl?
    # ys = sum(FF);
    # [entropy,bal, SDCS, RME] = BalanceEvl(nCluster, ys);
    
    # MATLAB code:
    # nCluster = length(unique(Y));
    # ... FF loops using 'y' ...
    # Wait, 'y' is the first argument in MATLAB function signature?
    # function [res]= my_eval_y(y,Y)
    # But usually y is truth, Y is pred? Or vice versa?
    # Context: best_map(Y,y). If Y is L1 (ref), y is L2 (to be permuted).
    # Usually we permute Prediction to match Truth? Or Truth to match Prediction?
    # ACC is usually sum(y_true == map(y_pred)).
    # If best_map(L1, L2) permutes L2 to match L1.
    # If we call best_map(Y, y), we permute y to match Y.
    # Then acc = mean(Y == newIndx). Pred == Permuted_Truth. This works.
    
    # BalanceEvl usage:
    # nCluster = length(unique(Y));
    # FF uses y(iSmp). y is Truth?
    # ys = sum(FF). Distribution of Truth?
    # This seems to evaluate the balance of 'y' (Truth) but using k from 'Y' (Pred).
    # If 'y' is truth, we are evaluating the balance of ground truth clusters?
    # Or maybe 'y' and 'Y' are swapped in meaning?
    # Let's stick to the code logic.
    
    uY = np.unique(Y)
    nCluster = len(uY)
    nSmp = len(Y)
    
    # Construct FF matrix based on y
    # We need to map y to 0..k indices to put in matrix
    uy, inv_y = np.unique(y, return_inverse=True)
    # Note: if y has more clusters than Y, or different, this matrix size might be issue if logic assumes match.
    # MATLAB: FF = zeros(nSmp, nCluster).
    # FF(iSmp, y(iSmp))=1. 
    # If y(iSmp) > nCluster, MATLAB expands matrix or errors? 
    # MATLAB expands automatically.
    # But we are in Python.
    # We should infer what 'y' represents.
    # If y is ground truth, its labels might be anything.
    # The MATLAB code implies y contains indices 1..nCluster?
    # We will compute the counts of y.
    
    # Simply: distribution of y
    # unique y and counts
    # But we need to match the logic: "ys = sum(FF)"
    # If FF size is (nSmp, nCluster), and we mark cols based on y.
    # Then ys is the count of each label in y.
    # But capped/buckets by nCluster?
    # If y has labels 1..K, and nCluster=K.
    # We will just compute counts of y.
    
    # However, for safe porting, let's replicate "ys" as counts of y's labels.
    # But we need to ensure we have 'nCluster' counts?
    # If y has different number of clusters than Y, BalanceEvl might get different vector size.
    # MATLAB: [entropy, ... ] = BalanceEvl(nCluster, ys).
    # BalanceEvl uses 'k' (first arg) for loops.
    # If ys has more/less elements, it might be issue.
    # Let's compute counts of y.
    vals, counts = np.unique(y, return_counts=True)
    ys = counts
    
    # If length(ys) != nCluster, what does MATLAB do?
    # It passes 'ys' (vector) and 'nCluster' (scalar).
    # BalanceEvl uses nCluster as 'k'.
    # loops i=1:k. Accesses N_cluster(i).
    # So ys must have at least k elements.
    # We should align ys to be length nCluster.
    # If len(ys) < nCluster, pad with 0.
    # If len(ys) > nCluster, truncate? (Unlikely if Y attempts to cluster y data).
    
    if len(ys) < nCluster:
        ys = np.concatenate([ys, np.zeros(nCluster - len(ys))])
    
    entropy, bal, SDCS, RME = balance_evl(nCluster, ys)
    
    res = [acc, nmi, purity, AR, RI, MI, HI, fscore, precision, recall, entropy, SDCS, RME, bal]
    return res
