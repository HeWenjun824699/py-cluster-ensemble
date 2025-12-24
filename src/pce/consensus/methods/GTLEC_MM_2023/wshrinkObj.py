import numpy as np

def soft(x, tau):
    """
    Soft thresholding operator.
    """
    return np.maximum(x - tau, 0)

def wshrinkObj(x, rho, sX, isWeight, mode=1):
    """
    Python implementation of wshrinkObj.m.
    This function solves the shrinkage problem for tensor factorization.

    x: flattened tensor data
    rho: parameter
    sX: shape of tensor [n1, n2, n3]
    isWeight: boolean flag
    mode: tensor orientation mode (1, 2, or 3)
    """

    C = 0
    if isWeight == 1:
        # sX is [d1, d2, d3]. In MATLAB, sX(3) is d3, sX(2) is d2.
        C = np.sqrt(sX[2] * sX[1])

    X = x.reshape(sX, order='F')

    #
    # --- Mode-based Permutation ---
    #
    if mode == 1:
        # Corresponds to MATLAB: Y=X2Yi(X,3);
        # Based on context from similar projects, X2Yi often performs a permutation.
        # Given the inverse is Yi2X(Y,3), this is likely a specific permutation.
        # Without its definition, we assume identity for now, as mode 3 is used by GTLEC.
        # This part might need correction if other modes are used.
        Y = X
    elif mode == 3:
        # Corresponds to MATLAB: Y=shiftdim(X, 1);
        # This shifts dimensions left: [d1, d2, d3] -> [d2, d3, d1]
        Y = np.transpose(X, (1, 2, 0))
    else: # mode == 2 or other
        Y = X

    #
    # --- FFT and Dimension Setup ---
    #
    Yhat = np.fft.fft(Y, axis=2)
    objV = 0

    if mode == 1:
        n3 = sX[1]
    elif mode == 3:
        n3 = sX[0]
    else:
        n3 = sX[2]

    #
    # --- Process FFT Slices ---
    # The MATLAB code has an if/else on isinteger(n3/2) which always evaluates
    # to false because n3 is a double. So we only implement the `else` block.
    #
    # Loop from i=1 to floor(n3/2)+1 in MATLAB
    # Python equivalent: range( (n3 // 2) + 1 )
    #
    for i in range((n3 // 2) + 1):
        # SVD on the i-th slice
        try:
            U, S, Vh = np.linalg.svd(Yhat[:, :, i], full_matrices=False)
        except np.linalg.LinAlgError:
            # If SVD fails, use a pseudo-inverse or return zeros
            U, S, Vh = np.linalg.svd(Yhat[:, :, i] + 1e-6 * np.eye(Yhat.shape[0]), full_matrices=False)


        # Apply soft thresholding
        if isWeight:
            # Adding epsilon for stability
            weight = C / (S + np.finfo(float).eps)
            tau = rho * weight
            S_new = soft(S, tau)
        else:
            tau = rho
            S_new = np.maximum(S - tau, 0)

        # Update objective function value
        objV += np.sum(S_new)

        # Reconstruct the slice
        Yhat[:, :, i] = U @ np.diag(S_new) @ Vh

        # Update the conjugate symmetric slice
        # In MATLAB loop: `if i > 1` (since i=1 is DC)
        # In Python loop: `if i > 0`
        # and not the Nyquist frequency (which is its own conjugate)
        if i > 0 and i != n3 / 2:
            # MATLAB index for conjugate is `n3-i+2`.
            # Python equivalent: `n3 - (i+1) + 1 = n3 - i`.
            conj_idx = n3 - i
            Yhat[:, :, conj_idx] = np.conj(U) @ np.diag(S_new) @ np.conj(Vh)
            
            # The original MATLAB code adds the objective value again for the
            # conjugate part. This is unusual but we replicate it.
            objV += np.sum(S_new)

    #
    # --- Inverse FFT and Reshape ---
    #
    Y = np.fft.ifft(Yhat, axis=2)

    # The result should be real if input was real and symmetry was preserved
    Y = np.real(Y)

    # Inverse permutation based on mode
    if mode == 1:
        # Corresponds to MATLAB: X = Yi2X(Y,3);
        X = Y # Assuming identity for now
    elif mode == 3:
        # Inverse of shiftdim(X, 1) is shiftdim(Y, 2)
        # Y is [d2, d3, d1]. We want [d1, d2, d3].
        # Transpose from (d2, d3, d1) -> (d1, d2, d3) is (2, 0, 1)
        X = np.transpose(Y, (2, 0, 1))
    else:
        X = Y

    x_out = X.flatten(order='F')
    return x_out, objV
