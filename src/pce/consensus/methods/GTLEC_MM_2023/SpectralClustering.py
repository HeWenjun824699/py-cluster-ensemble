import numpy as np
from sklearn.cluster import KMeans

def SpectralClustering(CKSym, n):
    """
    CKSym: Affinity matrix
    n: number of clusters
    """
    # N = size(CKSym,1);
    N = CKSym.shape[0]
    MAXiter = 1000
    REPlic = 20
    
    # DN = diag( 1./sqrt(sum(CKSym)+eps) );
    # sum(CKSym) sums columns in MATLAB. Here symmetric so rows/cols same.
    d = np.sum(CKSym, axis=0)
    dn_diag = 1.0 / np.sqrt(d + np.finfo(float).eps)
    DN = np.diag(dn_diag)
    
    # LapN = speye(N) - DN * CKSym * DN;
    # Use dense matrices as per MATLAB code 'svd' usage implies dense or converting.
    # MATLAB: speye is sparse, but svd works on it? Or converts?
    # Given size, dense is fine.
    LapN = np.eye(N) - DN @ CKSym @ DN
    
    # [uN,sN,vN] = svd(LapN);
    # MATLAB svd returns U, S, V where A = U*S*V'.
    # Python svd returns U, S, Vh where A = U*diag(S)*Vh.
    # Note: V in MATLAB is Vh.T in Python.
    
    U, S, Vh = np.linalg.svd(LapN)
    vN = Vh.T
    
    # kerN = vN(:,N-n+1:N);
    # MATLAB indices 1-based. N-n+1 to N.
    # e.g. N=100, n=3. 98, 99, 100. (Last 3 columns).
    # Python indices: 97, 98, 99. (Last 3 columns).
    # vN[:, -n:]
    kerN = vN[:, -n:]
    
    # for i = 1:N
    #    kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
    # end
    # Normalize rows
    row_norms = np.linalg.norm(kerN, axis=1, keepdims=True) + np.finfo(float).eps
    kerNS = kerN / row_norms
    
    # groups = kmeans(kerNS,n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    kmeans = KMeans(n_clusters=n, max_iter=MAXiter, n_init=REPlic).fit(kerNS)
    groups = kmeans.labels_
    
    # MATLAB labels are usually 1-based?
    # Python 0-based.
    # The caller expects labels.
    
    return groups
