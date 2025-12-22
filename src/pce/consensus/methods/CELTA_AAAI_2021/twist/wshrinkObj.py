import numpy as np
from scipy.linalg import svd
from .X2Yi import X2Yi
from .Yi2X import Yi2X
from .Trans_Faces import Trans_Faces # Imported but used inside X2Yi/Yi2X

# Helper to mimic shiftdim locally if needed, but we implemented it in X2Yi/Yi2X
# We will duplicate shiftdim here or import it if we want to be clean, 
# but wshrinkObj uses shiftdim directly in one branch.
# To avoid circular imports or duplication, let's just implement shiftdim helper inside wshrinkObj or use the one from numpy if valid.
# But numpy doesn't have shiftdim directly.
# Let's just define it inline or assume X2Yi's version is available if we refactor.
# For now, I will include a local shiftdim.

def shiftdim(X, n):
    if n == 0:
        return X
    ndim = X.ndim
    n = n % ndim
    axes = list(range(ndim))
    new_axes = axes[n:] + axes[:n]
    return np.transpose(X, new_axes)

def soft(s, tau_diag):
    """
    Soft thresholding operator.
    shat = max(shat - tau, 0) in the code logic for non-weighted.
    But for weighted: shat = soft(shat, diag(tau))
    
    In Matlab: soft(x, T) usually means sign(x).*max(abs(x) - T, 0).
    Here 's' seems to be singular values (always positive).
    And tau_diag is the diagonal of the threshold matrix?
    
    The code says:
    weight = C./(diag(shat) + eps);
    tau = rho*weight;
    shat = soft(shat, diag(tau));
    
    If 'shat' is the diagonal matrix of singular values (as returned by svd),
    then diag(shat) gets the vector of singular values.
    
    Wait, Matlab's svd: [U, S, V] = svd(A). S is diagonal matrix.
    diag(S) -> vector of singular values.
    
    The code:
    tau = rho * weight (vector)
    shat = soft(shat, diag(tau))
    
    If shat is a diagonal matrix, and diag(tau) is a diagonal matrix of thresholds.
    Then soft(shat, diag(tau)) likely applies soft thresholding element-wise on the diagonal.
    
    Result is max(S - Tau, 0).
    """
    # Assuming s is diagonal matrix and tau_diag is diagonal matrix
    # Extract diagonals
    s_vec = np.diag(s)
    tau_vec = np.diag(tau_diag)
    
    # Apply threshold
    res_vec = np.maximum(s_vec - tau_vec, 0)
    
    # Reconstruct diagonal matrix
    return np.diag(res_vec)

def wshrinkObj(x, rho, sX, isWeight, mode=1):
    """
    Weighted shrinkage operator for tensor.
    
    Args:
        x: Flattened tensor data (column-major/Fortran order from Matlab)
        rho: Parameter
        sX: Shape of tensor [n1, n2, n3]
        isWeight: Boolean/Int flag
        mode: 1, 2, or 3
    
    Returns:
        x: Processed flattened tensor
        objV: Objective value
    """
    
    # if isWeight == 1
    #     C = sqrt(sX(3)*sX(2));
    # end
    if isWeight == 1:
        # Note: Matlab indices sX(3), sX(2). Python 0-based: sX[2], sX[1].
        C = np.sqrt(sX[2] * sX[1])
    else:
        C = 0 # Undefined in Matlab code if isWeight!=1, but probably not used.
    
    # if ~exist('mode','var') -> default mode=1
    # Handled by default arg.
    
    # X=reshape(x,sX);
    # Matlab reshape is column-major.
    X = x.reshape(sX, order='F')
    
    # if mode == 1
    #     Y=X2Yi(X,3);
    # elseif mode == 3
    #     Y=shiftdim(X, 1);
    # else
    #     Y = X;
    # end
    if mode == 1:
        Y = X2Yi(X, 3)
    elif mode == 3:
        Y = shiftdim(X, 1)
    else:
        Y = X
        
    # Yhat = fft(Y,[],3);
    # Matlab: fft(Y, [], 3) -> FFT along 3rd dimension (index 3 -> Python index 2)
    Yhat = np.fft.fft(Y, axis=2)
    
    objV = 0
    
    # if mode == 1
    #     n3 = sX(2);
    # elseif mode == 3
    #     n3 = sX(1);
    # else
    #     n3 = sX(3);
    # end
    if mode == 1:
        n3 = sX[1]
    elif mode == 3:
        n3 = sX[0]
    else:
        n3 = sX[2]
        
    # if isinteger(n3/2) ...
    # Matlab isinteger checks data type, not value parity. But usually used for parity check concept?
    # NO. isinteger(3.0) is false. isinteger(int8(3)) is true.
    # But here n3 comes from sX (dimensions), usually double in Matlab unless cast.
    # Wait, 'isinteger(n3/2)' in Matlab checks if the RESULT is of integer TYPE.
    # 5/2 = 2.5 (double). 4/2 = 2 (double).
    # So isinteger(n3/2) is ALWAYS FALSE for standard double calculations in Matlab!
    # UNLESS n3/2 returns an integer type? No, division usually returns double.
    # Maybe the author meant `mod(n3, 2) == 0`?
    # Let's check the code logic.
    # if isinteger(n3/2) ... else ...
    # In the else branch, it iterates to endValue = int16(n3/2+1).
    # If n3 is even (e.g. 4), n3/2+1 = 3.
    # If n3 is odd (e.g. 3), n3/2+1 = 2.5 -> int16(2.5) = 3 (rounding to nearest).
    
    # IMPORTANT: Many Matlab tensor codes use a specific loop structure for FFT property (conjugate symmetry).
    # If n3 is even, we handle 1, n3/2+1 (Nyquist), and pairs in between.
    # If n3 is odd, we handle 1, and pairs in between.
    # The condition `isinteger(n3/2)` is likely a bug in the original Matlab code or I am misinterpreting it.
    # However, usually the check is `mod(n3, 2) == 0`.
    # Let's look at the 'else' block:
    # endValue = int16(n3/2+1).
    # It assumes n3 is odd?
    # If n3=3, endValue=3. Loop 1 to 3. 
    # i=1. i=2 (updates n3-2+2=3). i=3 (updates n3-3+2=2).
    # Overlap?
    
    # Standard tensor SVD via FFT (twist):
    # For real tensor X, fft(X) has symmetry.
    # Yhat(:,:,i) and Yhat(:,:,n3-i+2) are conjugates.
    # We only need to compute SVD for first ceil((n3+1)/2) slices.
    
    # Let's trust the logic structure based on standard practice if the code is ambiguous.
    # "isinteger" in Matlab is indeed type check. 
    # If sX contains integers (e.g. from size()), sX(i) is double.
    # So isinteger(sX(i)/2) is FALSE.
    # So it likely ALWAYS goes to the 'else' block in the original code?
    # Let's verify if sX passed to function is integer class.
    # In TensorEnsemble.m: sX = [n, n, 2]; (Double by default).
    # So it goes to 'else'.
    
    # BUT, let's look at the 'if' block. It handles `endValue+1` explicitly outside loop.
    # This suggests handling the Nyquist frequency (middle element) separately, which implies Even n3.
    # So the 'if' block is INTENDED for Even numbers.
    # The 'else' block is INTENDED for Odd numbers (or the author's fallback).
    # Given the likely bug `isinteger` instead of `mod`, I will assume `n3 % 2 == 0` triggers the first block.
    
    # Logic for loop bounds and execution path
    # In Matlab: if isinteger(n3/2) ... else ...
    # Since sX contains doubles, n3/2 is a double. isinteger(double) is False.
    # Therefore, Matlab ALWAYS executes the 'else' block.
    # We strictly replicate the 'else' block logic here.
    
    # endValue = int16(n3/2+1);
    # Matlab int16 rounds to nearest integer (ties away from zero for positive).
    # Python round() rounds to nearest even for ties.
    # We use int(n3/2 + 1 + 0.5) logic for standard rounding half-up behavior on positive numbers.
    val = n3 / 2 + 1
    endValue = int(np.floor(val + 0.5))
    
    # for i = 1:endValue
    for i in range(1, endValue + 1):
        # Matlab 1-based indexing
        slice_idx = i - 1
        
        # [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
        U, s, Vh = svd(Yhat[:,:,slice_idx], full_matrices=False)
        
        # Construct diagonal matrix S
        S_mat = np.diag(s)
        
        if isWeight:
            # weight = C./(diag(shat) + eps);
            weight = C / (s + eps)
            tau = rho * weight
            # shat = soft(shat,diag(tau));
            S_mat = soft(S_mat, np.diag(tau))
            s_new = np.diag(S_mat)
        else:
            tau = rho
            # shat = max(shat - tau,0);
            S_mat = np.maximum(S_mat - tau, 0)
            s_new = np.diag(S_mat)
            
        # objV = objV + sum(shat(:));
        objV += np.sum(s_new)
        
        # Yhat(:,:,i) = uhat*shat*vhat';
        Yhat[:,:,slice_idx] = U @ S_mat @ Vh
        
        if i > 1:
            # Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            # Matlab index: n3 - i + 2
            mat_pair_idx = n3 - i + 2
            py_pair_idx = mat_pair_idx - 1
            
            # Check strictly if index is within bounds (Matlab would error if not, so we assume valid)
            if py_pair_idx < n3:
                # conj(uhat)*shat*conj(vhat)'
                # conj(vhat)' -> V^T (Transpose of V). 
                # Vh is V^H. Vh.conj() is V^T.
                Yhat[:,:,py_pair_idx] = U.conj() @ S_mat @ Vh.conj()
                
                # objV = objV + sum(shat(:));
                objV += np.sum(s_new)

    # Y = ifft(Yhat,[],3);
    # axis=2
    Y = np.fft.ifft(Yhat, axis=2)
    
    # if mode == 1
    #     X = Yi2X(Y,3);
    # elseif mode == 3
    #     X = shiftdim(Y, 2);
    # else
    #     X = Y;
    # end
    if mode == 1:
        X = Yi2X(Y, 3)
    elif mode == 3:
        X = shiftdim(Y, 2)
    else:
        X = Y
        
    # x = X(:);
    x = X.flatten(order='F')
    
    return x, objV
