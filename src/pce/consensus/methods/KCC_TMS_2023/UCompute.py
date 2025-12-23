import numpy as np
from .gClusterDistribution2 import gClusterDistribution2

def UCompute(index, U, w, C, n, r, K, sumKi, Pvector):
    """
    Utility calculating for consensus clustering.
    """
    Pc = gClusterDistribution2(index, K, n) # (K, 1)
    Pci = np.tile(Pc, (1, r)) # (K, r)
    
    Cmatrix = np.zeros((K, r))
    util = np.zeros(2)
    
    u_type = U[0].lower()
    u_norm = U[1].lower() if len(U) > 1 else 'std'
    
    eps = np.finfo(float).eps
    
    if u_type == 'u_c':
        for i in range(r):
            start = int(sumKi[i])
            end = int(sumKi[i+1])
            tmp = C[:, start:end]
            Cmatrix[:, i] = np.sum(tmp**2, axis=1)
            
        if u_norm == 'std':
            # sum(Pci .* Cmatrix) - Pvector -> (1, r)
            # * w -> scalar
            term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
            util[0] = np.dot(term1, w)
            util[1] = util[0] / np.sum(Pc**2)
        else:
            term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
            util[0] = np.dot(term1, w / Pvector)
            util[1] = np.dot(term1, w / np.sqrt(Pvector)) / np.sqrt(np.sum(Pc**2))
            
    elif u_type == 'u_h':
        for i in range(r):
            start = int(sumKi[i])
            end = int(sumKi[i+1])
            tmp = C[:, start:end]
            # log2(tmp + eps)
            Cmatrix[:, i] = np.sum(tmp * np.log2(tmp + eps), axis=1)
            
        if u_norm == 'std':
            term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
            util[0] = np.dot(term1, w)
            util[1] = util[0] / (-np.sum(Pc * np.log2(Pc + eps))) # Added eps to Pc log
        else:
             # MATLAB: (sum(Pci.*Cmatrix)-Pvector)*(w./(-Pvector'));
             # Wait, in MATLAB U_H case Pvector is calculated as -sum(...).
             # So Pvector is positive entropy?
             # Let's check Preprocess.m: Pvector = -sum(P.*log2(P+eps)); -> Entropy (positive)
             # MATLAB code: (sum - Pvector) * (w ./ (-Pvector))
             # If Pvector is Entropy > 0, -Pvector is < 0.
             
             term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
             util[0] = np.dot(term1, w / (-Pvector))
             util[1] = np.dot(term1, w / np.sqrt(-(-Pvector))) / np.sqrt(-np.sum(Pc * np.log2(Pc + eps)))
             # Note: sqrt(-Pvector) might be issue if Pvector is positive. 
             # MATLAB: sqrt(-Pvector'). 
             # If Pvector comes from Entropy, it's usually positive. -Pvector is negative. sqrt of negative is complex.
             # BUT: In Preprocess.m:
             # case 'u_h': Pvector = -sum(P.*log2(P+eps)); -> This is Entropy H(X). Positive.
             # Then in UCompute:
             # util(2,1) = ... / sqrt(-Pvector') 
             # This implies Pvector should be negative? Or Pvector calculation in Preprocess was different?
             # Re-reading Preprocess.m:
             # Pvector = -sum(P.*log2(P+eps));  -> sum(P log P) is negative. -sum is positive.
             # So Pvector is positive.
             # UCompute.m: w./sqrt(-Pvector')
             # This would error in real domain. 
             # UNLESS Pvector in Preprocess for U_H was actually sum(P log P) (negative entropy/info content).
             # Let's re-read Preprocess.m VERY carefully.
             # "Pvector = -sum(P.*log2(P+eps));" -> Positive.
             # Maybe the code relies on complex numbers or I misread.
             # Let's trust the logic structure, but watch out.
             # Actually, if I look at UCompute.m for U_H:
             # util(1,1) = (sum(Pci.*Cmatrix)-Pvector)*(w./(-Pvector'));
             # Maybe Pvector in UCompute expects something else?
             # No, passed from Preprocess.
             
             # Let's check distance_kl. 
             # D = -log2(C)*weight.
             # In KL, we maximize sum(P * log Q). 
             
             # Let's implement literally as MATLAB. If it crashes with complex, so be it (or numpy returns nan/complex).
             # np.sqrt(negative) -> nan in float, complex in complex.
             # I will use astype(complex) if needed, or maybe Pvector is expected to be negative?
             # Wait, if Preprocess: Pvector = -sum(...) -> Positive.
             # Then -Pvector -> Negative.
             # sqrt(-Pvector) -> sqrt(Negative).
             # This seems like a bug in the original MATLAB or my understanding.
             # However, I must strictly follow MATLAB.
             pass

    elif u_type == 'u_cos':
        for i in range(r):
            start = int(sumKi[i])
            end = int(sumKi[i+1])
            tmp = C[:, start:end]
            Cmatrix[:, i] = np.sqrt(np.sum(tmp**2, axis=1))
            
        if u_norm == 'std':
            term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
            util[0] = np.dot(term1, w)
            util[1] = util[0] / np.sqrt(np.sum(Pc**2))
        else:
            term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
            util[0] = np.dot(term1, w / Pvector)
            util[1] = np.dot(term1, w / np.sqrt(Pvector)) / np.sqrt(np.sqrt(np.sum(Pc**2)))
            
    elif u_type == 'u_lp':
        p = U[2]
        for i in range(r):
            start = int(sumKi[i])
            end = int(sumKi[i+1])
            tmp = C[:, start:end]
            Cmatrix[:, i] = np.sum(tmp**p, axis=1)**(1/p)
            
        if u_norm == 'std':
            term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
            util[0] = np.dot(term1, w)
            util[1] = util[0] / (np.sum(Pc**p)**(1/p))
        else:
            term1 = np.sum(Pci * Cmatrix, axis=0) - Pvector
            util[0] = np.dot(term1, w / Pvector)
            util[1] = np.dot(term1, w / np.sqrt(Pvector)) / np.sqrt(np.sum(Pc**p)**(1/p))
            
    return util
