import numpy as np

def select_pairs(W, S, batchsize, n):
    """
    Select pairs to label based on W (uncertainty/informativeness) and S (similarity).
    
    Parameters:
    W : numpy.ndarray
        Weight matrix (or uncertainty matrix).
    S : numpy.ndarray
        Similarity matrix.
    batchsize : int
    n : int
        Number of samples.
        
    Returns:
    labeled : numpy.ndarray
        Indices of selected pairs (linear indices).
    """
    # MATLAB: [~,idx_tolabel]=sort(W(:),'ascend');
    # Flatten W
    W_flat = W.flatten(order='F') # 'F' for column-major matching MATLAB linear indexing? 
    # Actually Python is usually row-major. Let's stick to default 'C' but remember 
    # ind2sub later needs to match.
    # W is symmetric usually, so order might not matter for values, 
    # but for index mapping it does.
    # MATLAB stores column-major.
    # If we want to return linear indices compatible with MATLAB logic (though Python uses tuples),
    # we should probably return (row, col) indices or keep 1D index consistent.
    # Let's use flattened row-major for Python consistency unless 'n' implies specific shape.
    # The input W is (n, n).
    
    W_flat = W.flatten() # Row-major default
    
    idx_tolabel = np.argsort(W_flat) # Ascending
    
    # MATLAB uses 1-based indexing for batchsize retrieval
    # threshold=W(idx_tolabel(batchsize));
    # In Python, index is batchsize-1
    threshold_idx = idx_tolabel[batchsize - 1]
    threshold = W_flat[threshold_idx]
    
    # idx_tolabel1=find(W<=threshold+eps);
    # In MATLAB, find returns linear indices.
    eps = np.finfo(float).eps
    idx_tolabel1 = np.where(W_flat <= threshold + eps)[0]
    
    max_idx = len(idx_tolabel1)
    
    # MATLAB: if threshold==W(idx_tolabel(1))
    if threshold == W_flat[idx_tolabel[0]]:
        # [i_cand,j_cand]=ind2sub([n,n],idx_tolabel1);
        # Python unravelling
        i_cand, j_cand = np.unravel_index(idx_tolabel1, (n, n))
        
        score = np.zeros(max_idx)
        for sel in range(max_idx):
            # score(sel)=sum(S(i_cand(sel),:))*sum(S(j_cand(sel),:));
            score[sel] = np.sum(S[i_cand[sel], :]) * np.sum(S[j_cand[sel], :])
            
        # [~,idx_tolabel2]=sort(score,'descend');
        idx_tolabel2 = np.argsort(score)[::-1]
        
        # labeled=idx_tolabel1(idx_tolabel2(1:batchsize));
        # Take top batchsize from the sorted candidates
        if batchsize > len(idx_tolabel2):
            bs = len(idx_tolabel2)
        else:
            bs = batchsize
            
        labeled = idx_tolabel1[idx_tolabel2[:bs]]
    else:
        # labeled=idx_tolabel(1:batchsize);
        labeled = idx_tolabel[:batchsize]
        
    return labeled
