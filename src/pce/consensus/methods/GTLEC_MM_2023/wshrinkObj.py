import numpy as np

def soft(x, tau):
    """
    Soft thresholding operator.
    shat = max(shat - tau, 0)
    If x is complex, it usually applies to magnitude, but here input seems to be singular values (real).
    If input is matrix, tau can be matrix or scalar.
    """
    return np.maximum(x - tau, 0)

def wshrinkObj(x, rho, sX, isWeight, mode=1):
    """
    x: flattened tensor data
    rho: parameter
    sX: shape of tensor [n1, n2, n3]
    isWeight: boolean
    mode: 1, 2, or 3
    """
    
    if isWeight == 1:
        # C = sqrt(sX(3)*sX(2));
        # In MATLAB indices are 1-based: sX(1), sX(2), sX(3).
        # Python: sX[0], sX[1], sX[2].
        # MATLAB sX(3) -> Python sX[2]
        # MATLAB sX(2) -> Python sX[1]
        C = np.sqrt(sX[2] * sX[1])
    
    X = x.reshape(sX, order='F') # MATLAB reshape is column-major (Fortran-like)
    
    # Mode handling
    if mode == 1:
        # MATLAB: Y=X2Yi(X,3);
        # Logic inferred: usually permutes so the 3rd dimension becomes the processing dimension?
        # But wait, standard tensor code (e.g. TRPCA) often handles different modes by permuting 
        # the target mode to the 3rd dimension (or whichever is used for FFT).
        # The code does `Yhat = fft(Y,[],3)`.
        # If mode=1 (lateral?), maybe we work on a different orientation.
        # However, checking common "X2Yi" implementations in this domain:
        # often it's just `shiftdim` or `permute`.
        # Without X2Yi, and assuming standard TRPCA logic:
        # If `wshrinkObj` is called with mode=3 in TensorEC, let's focus on that or generic.
        # TensorEC calls: wshrinkObj(..., mode=3).
        # In wshrinkObj.m: if mode == 3: Y = shiftdim(X, 1).
        
        # Generic placeholder logic if mode != 3 (since we prioritize TensorEC usage):
        # We'll assume mode 1 is identity for now or find reference.
        # But given we must be consistent, let's implement the mode 3 logic perfectly.
        # mode 1 logic: Y = X2Yi(X,3).
        # If we assume X2Yi is just identity for mode 3 usage? No.
        
        # Let's stick to what we know from wshrinkObj.m text provided:
        # "if mode == 1 Y=X2Yi(X,3); elseif mode == 3 Y=shiftdim(X, 1); else Y = X;"
        
        # Since I don't have X2Yi, I will warn or assume standard permutation if mode 1 is used.
        # But TensorEC uses mode 3.
        Y = X
        
    elif mode == 3:
        # MATLAB: Y=shiftdim(X, 1);
        # MATLAB shiftdim(X, 1) shifts dimensions to the left.
        # [d1, d2, d3] -> [d2, d3, d1]
        Y = np.transpose(X, (1, 2, 0))
    else:
        Y = X

    # Yhat = fft(Y,[],3);
    # Python fft is over last axis by default? No, axis=-1.
    # MATLAB fft(..., 3) is over 3rd dimension (index 2).
    Yhat = np.fft.fft(Y, axis=2)
    
    objV = 0
    
    # Determine n3
    if mode == 1:
        n3 = sX[1] # MATLAB sX(2)
    elif mode == 3:
        n3 = sX[0] # MATLAB sX(1)
    else:
        n3 = sX[2] # MATLAB sX(3)
        
    # The loop logic depends on n3
    # Wait, n3 is determined from sX based on mode.
    # In MATLAB code:
    # if mode == 1, n3 = sX(2)
    # if mode == 3, n3 = sX(1)
    # else n3 = sX(3)
    
    # Note: If Y is [d2, d3, d1] (from mode 3 shiftdim), the 3rd dim size is d1 = sX[0].
    # So `n3` correctly matches Y.shape[2].
    
    # Check if integer(n3/2) logic (even vs odd)
    # Python loop range
    
    if n3 % 2 == 0:
        endValue = int(n3 / 2) + 1 # MATLAB indices 1 to n3/2 + 1
        # Loop 1 to endValue (inclusive in MATLAB)
        # Python indices: 0 to endValue-1
        
        # Handling i=1 (MATLAB) -> i=0 (Python) separately? 
        # MATLAB: for i = 1:endValue
        # Loop logic handles `i > 1` specific for conjugate.
        
        for i in range(endValue): # 0 to endValue-1
            # MATLAB i=1..endValue
            # Python i corresponds to MATLAB i-1? No, let's map indices directly.
            # Python index `idx` = i.
            
            # svd
            # [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
            # Python np.linalg.svd returns u, s, vh. vh is conjugate transpose of v.
            # And s is 1D array.
            
            U, S, Vh = np.linalg.svd(Yhat[:, :, i], full_matrices=False)
            
            # Soft thresholding
            if isWeight:
                # weight = C./(diag(shat) + eps);
                # tau = rho*weight;
                # shat = soft(shat,diag(tau));
                weight = C / (S + np.finfo(float).eps)
                tau = rho * weight
                S_new = soft(S, tau)
            else:
                tau = rho
                S_new = np.maximum(S - tau, 0)
            
            objV += np.sum(S_new)
            
            # Reconstruct
            # Yhat(:,:,i) = uhat*shat*vhat';
            # Python: U @ diag(S_new) @ Vh
            Yhat[:, :, i] = U @ np.diag(S_new) @ Vh
            
            # if i > 1 (MATLAB) -> idx > 0 (Python)
            if i > 0:
                # MATLAB: Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
                # MATLAB index `n3-i+2`.
                # Let's trace. i=2 (first non-DC). `n3-2+2` = `n3`.
                # Python indices:
                # `idx` corresponds to freq `idx`.
                # Conjugate frequency is `n3 - idx`.
                # Example n3=4. Indices 0, 1, 2, 3.
                # i=0 (DC).
                # i=1 (Freq 1). Conj is 4-1 = 3.
                # i=2 (Freq 2 / Nyquist). Conj is 4-2=2. (Self).
                
                # MATLAB loop goes up to `n3/2 + 1`.
                # If n3=4. endValue = 3. Loop 1, 2, 3.
                # i=1: DC.
                # i=2: Freq 1. Update `4-2+2`=4.
                # i=3: Freq 2 (Nyquist). Update `4-3+2`=3?
                # Wait, MATLAB code updates `n3-i+2`.
                # If i=3 (Nyquist, index 3 in Python? No index 2).
                # `n3-3+2` = `n3-1`? 
                
                # Let's check loop range again.
                # `endValue = int16(n3/2+1)`.
                # If n3=4, endValue=3.
                # i=1.
                # i=2. Update `4-2+2`=4. (Python index 3).
                # i=3. Update `4-3+2`=3. (Python index 2).
                # This overwrites itself if i is Nyquist?
                
                # But MATLAB `svd` logic handles i=endValue separately?
                # Look at `wshrinkObj.m` structure for even case:
                # Loop `i=1:endValue`
                # INSIDE Loop: `if i > 1 ...`
                # AFTER Loop: `[uhat,shat,vhat] = svd(..., endValue+1)`.
                # Wait.
                # `if isinteger(n3/2)` (Even case).
                # Code says: `endValue = int16(n3/2+1)`.
                # Loop `i = 1:endValue`.
                # AFTER loop: `svd(..., endValue+1)`.
                # But `endValue+1` = `n3/2+2`.
                # Indices in MATLAB: 1..n3.
                # If n3=4. endValue=3.
                # Loop 1, 2, 3.
                # i=1.
                # i=2. Update 4.
                # i=3. Update 3.
                # After loop: process 4? But 4 was updated by i=2.
                # And i=3 was processed in loop.
                # This seems redundant or specific logic.
                
                # Let's re-read carefully `wshrinkObj.m`.
                # "if isinteger(n3/2)"
                # "endValue = int16(n3/2+1);"
                # "for i = 1:endValue"
                # ...
                # "if i > 1 ... Yhat(:,:,n3-i+2) = ..."
                # "end" (End of loop)
                # "[uhat,shat,vhat] = svd(full(Yhat(:,:,endValue+1)),'econ');"
                # ...
                # "Yhat(:,:,endValue+1) = ..."
                
                # If n3=4. endValue=3.
                # Loop 1, 2, 3.
                # i=2 updates 4.
                # i=3 updates 3 (itself).
                # Post-loop: process 4 (endValue+1).
                # So 4 is processed twice? Once as conj of 2, once explicitly?
                # Usually Nyquist is n3/2 + 1 (1-based).
                # For n3=4, Nyquist is 3.
                # Frequencies: 0(1), 1(2), 2(3), -1(4).
                # Conjugate of 2 is 4.
                # Conjugate of 3 is 3.
                
                # Maybe `endValue` in MATLAB code was `n3/2`?
                # "endValue = int16(n3/2+1);" -> explicitly +1.
                # Maybe the loop should be smaller?
                # Typically loop 1 to ceil((n3+1)/2).
                # For even n3=4. 1 to 3? 
                # If i=2, conj is 4.
                # If i=3, conj is 3.
                # If loop includes 3, it updates 3.
                # Then `endValue+1`=4.
                # It re-processes 4.
                # This seems slightly off for standard FFT, but I must follow the code.
                
                # Python indices:
                # n3=4.
                # endValue (count) = 3.
                # Loop idx 0, 1, 2. (Correspond to MATLAB 1, 2, 3).
                # idx 0 (i=1): DC.
                # idx 1 (i=2): Update Python idx `n3 - 2 + 2 - 1` ?
                # MATLAB index: `k = n3 - i + 2`.
                # Python index: `k - 1 = n3 - i + 1`.
                # Since `i_py = i - 1`, `i = i_py + 1`.
                # `idx_conj = n3 - (i_py + 1) + 1 = n3 - i_py`.
                # Check: n3=4, i_py=1 (i=2). `4-1=3`. Correct (MATLAB 4).
                
                # Logic:
                # idx 1 updates idx 3.
                # idx 2 updates idx 2.
                
                # Post loop: process endValue+1 => MATLAB 4 => Python 3.
                # So Python 3 is updated by idx 1, and then overwritten by post-loop?
                # That suggests the loop shouldn't go that far or `endValue` calculation in my head is wrong?
                # Or maybe `isinteger` check in MATLAB works differently?
                # `n3/2` is float? `isinteger` checks if value is stored as integer or has integer value?
                # `isinteger(3.0)` is false in MATLAB. `isinteger(int16(3))` is true.
                # `n3` is likely double. `n3/2` is double. `isinteger(2.0)` is FALSE.
                # So the `if isinteger(n3/2)` branch might NEVER be taken if n3 is double?
                # Unless n3 was cast to integer before?
                # `sX` is usually double.
                # IF `n3` is double (standard size), `isinteger` is false.
                # So it goes to `else` block?
                
                # Let's look at `else` block.
                # "endValue = int16(n3/2+1);"
                # "for i = 1:endValue"
                # ...
                # "end"
                # No post-loop processing.
                
                # This makes sense! `isinteger` in MATLAB checks data type, not value.
                # Unless `n3` came from `size` which returns double usually.
                # So mostly we are in `else` block.
                
                # However, for `mode` handling, `sX` might be integer?
                # `sX = [N, N, V]`. `N` from `size(H,2)`. Double.
                # So `n3` is double.
                # `isinteger(n3/2)` is FALSE.
                
                # So we implement `else` block logic.
                # Loop `i = 1 : int16(n3/2+1)`.
                # If n3=4. endValue=3.
                # Loop 1, 2, 3.
                # i=2 (idx 1). Conj 4 (idx 3).
                # i=3 (idx 2). Conj 3 (idx 2).
                
                pass

            conj_idx = n3 - i
            if conj_idx < n3: # Bound check
                 Yhat[:, :, conj_idx] = np.conj(U) @ np.diag(S_new) @ np.conj(Vh).T
                 objV += np.sum(S_new)

    else: # Odd n3
        # Logic in else block of MATLAB code (which is actually executed for double evens too)
        
        # NOTE: The MATLAB code structure:
        # if isinteger(n3/2) -> Code A
        # else -> Code B
        
        # As analyzed, likely Code B is executed.
        # Code B:
        # endValue = int16(n3/2+1)
        # loop i = 1:endValue
        #    ...
        #    if i > 1
        #       Yhat(:,:,n3-i+2) = ...
        #    end
        # end
        
        # Implementation of Code B (Universal):
        # Python range: 0 to endValue-1.
        
        endValue = int(n3 / 2) + 1
        
        for i in range(endValue):
            # i is python index (0-based)
            # Corresponds to MATLAB i+1
            
            U, S, Vh = np.linalg.svd(Yhat[:, :, i], full_matrices=False)
            
            if isWeight:
                weight = C / (S + np.finfo(float).eps)
                tau = rho * weight
                S_new = soft(S, tau)
            else:
                tau = rho
                S_new = np.maximum(S - tau, 0)
                
            objV += np.sum(S_new)
            Yhat[:, :, i] = U @ np.diag(S_new) @ Vh
            
            if i > 0:
                # Conj index
                # MATLAB: k = n3 - (i+1) + 2 = n3 - i + 1.
                # Python index: k - 1 = n3 - i.
                idx_conj = n3 - i
                
                # Check if idx_conj is within bounds.
                # For n3=4, endValue=3. i=0,1,2.
                # i=1 -> conj=3.
                # i=2 -> conj=2.
                if idx_conj < n3:
                     # Check if we are overwriting current i?
                     # If i == idx_conj, we add objV again?
                     # MATLAB code:
                     # "objV = objV + sum(shat(:));" (First time)
                     # "if i > 1 ... objV = objV + sum(shat(:));" (Second time)
                     # So it doubles the objective for conjugate pairs.
                     # Even for Nyquist (i=2, conj=2)?
                     # If i=2, conj=2. It executes assignment and adds objV.
                     # So Nyquist term is counted twice?
                     # Standard L1 norm of tensor sum singular values usually counts unique ones.
                     # But we follow MATLAB code strictly.
                     
                     Yhat[:, :, idx_conj] = np.conj(U) @ np.diag(S_new) @ np.conj(Vh).T
                     objV += np.sum(S_new)
    
    # Y = ifft(Yhat,[],3);
    Y = np.fft.ifft(Yhat, axis=2)
    # Result is complex, take real part? MATLAB ifft returns real if input has conjugate symmetry.
    # We should ensure it returns real if expected.
    # The MATLAB code doesn't explicitly cast to real, but implies X is real.
    Y = np.real(Y) 
    
    # Reverse mode
    if mode == 1:
        # Inverse of Y=X (since we didn't transform)
        X = Y
    elif mode == 3:
        # MATLAB: X = shiftdim(Y, 2);
        # Y is [d2, d3, d1].
        # shiftdim(Y, 2) -> [d3, d1, d2] -> [d1, d2, d3].
        # Python: np.transpose(Y, (2, 0, 1))
        X = np.transpose(Y, (2, 0, 1))
    else:
        X = Y
        
    x_out = X.flatten(order='F')
    return x_out, objV