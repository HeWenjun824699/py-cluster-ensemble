import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def run_ptgp_v2(base_cls, pts_sim, range_k):
    """
    Run PTGP (Probability Trajectory based Graph Partitioning).

    Args:
        base_cls: N x M matrix of cluster labels (microclusters).
        pts_sim: N x N similarity matrix (PTS).
        range_k: List of cluster numbers to generate.

    Returns:
        results: Clustering results (N x len(range_k)).
    """
    base_cls = base_cls.copy() # Ensure we don't modify original
    N, M = base_cls.shape

    # CRITICAL FIX 1: Convert PTS to sparse matrix to avoid memory explosion
    # when computing B = PTS * B. Dense * Sparse -> Dense (huge memory).
    # Sparse * Sparse -> Sparse (efficient).
    pts_sim = sparse.csr_matrix(pts_sim)

    # CRITICAL FIX 2: Robust Indexing / Overlap Prevention
    # Use max(col) + 1 to determine offsets. This works for both 0-based and 1-based labels.
    # It might leave some empty columns (e.g. index 0 if 1-based), but prevents
    # merging clusters from different partitions (Overlap Bug).
    max_cls = np.max(base_cls, axis=0)

    # If labels are 1-based, max is K. If we allocate K, next starts at K.
    # 1..K. Next starts K. Overlap at K.
    # So we need at least max+1 per column to be safe if 0-based (0..max)
    # or just max if 1-based?
    # Safer: max + 1. Graph size might be slightly larger (with zero cols), but correct.

    cnt_per_col = max_cls + 1
    cumulative_max = np.cumsum(cnt_per_col)
    cnt_cls = int(cumulative_max[-1])

    # Add offsets to columns 1 to end
    offsets = np.zeros(M, dtype=int)
    offsets[1:] = cumulative_max[:-1]

    base_cls_shifted = base_cls + offsets

    # Build Bipartite Graph B
    row_list = []
    col_list = []
    val_list = []

    for j in range(M):
        # Get labels for this column
        # Filter negative labels if any (noise)
        valid_mask = base_cls[:, j] >= 0

        if np.any(valid_mask):
            row_list.append(np.arange(N)[valid_mask])
            # Use shifted values directly.
            # If 0-based: 0 -> 0+offset.
            # If 1-based: 1 -> 1+offset.
            col_list.append(base_cls_shifted[valid_mask, j])
            val_list.append(np.ones(np.sum(valid_mask)))

    row_ind = np.concatenate(row_list)
    col_ind = np.concatenate(col_list)
    data = np.concatenate(val_list)

    B = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(N, cnt_cls))

    # Weight edges in B according to PTS similarity
    # Da = diag(1./sum(B)); -> sum(B) sums columns (1xCntCls)
    col_sums = np.array(B.sum(axis=0)).flatten()
    # Avoid div by zero
    col_sums[col_sums == 0] = 1e-10

    # Da is diagonal matrix of 1/col_sums
    Da = sparse.diags(1.0 / col_sums)
    B = B.dot(Da)

    # B = PTS_sim * B * Da (Wait, previous line was B*Da, now PTS_sim * (B*Da))
    # PTS_sim is now sparse, so this remains sparse.
    B = pts_sim.dot(B)

    results = np.zeros((N, len(range_k)), dtype=int)

    for i, n_seg in enumerate(range_k):
        try:
            results[:, i] = bipartite_graph_partitioning(B, n_seg)
        except Exception as e:
            # print(f"Error in PTGP for k={n_seg}: {e}")
            results[:, i] = 1

    return results

def bipartite_graph_partitioning(B, n_seg):
    """
    Partition the bipartite graph using Ncut.
    """
    Nx, Ny = B.shape
    if Ny < n_seg:
        raise ValueError("The cluster number is too large!")

    # dx = sum(B, 2) -> Row sums
    dx = np.array(B.sum(axis=1)).flatten()
    dx[dx == 0] = 1e-10

    # Dx = sparse diag 1./dx
    Dx = sparse.diags(1.0 / dx)

    # Wy = B' * Dx * B
    # All matrices are sparse, so Wy is sparse (typically much smaller than N if cnt_cls < N)
    # Actually size is cnt_cls x cnt_cls.
    Wy = B.T.dot(Dx).dot(B)

    # Normalized affinity matrix
    # d = sum(Wy, 2)
    d = np.array(Wy.sum(axis=1)).flatten()
    d[d == 0] = 1e-10 # Avoid div by zero

    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(d))

    # nWy = D * Wy * D
    nWy = D_inv_sqrt.dot(Wy).dot(D_inv_sqrt)

    # Symmetrize
    nWy = (nWy + nWy.T) / 2.0

    # Eigenvectors
    # Use eigh for symmetric/hermitian
    # Convert to dense since nWy is (cnt_cls x cnt_cls), usually reasonable size.
    vals, vecs = eigh(nWy.toarray())

    # Sort descending
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]

    # Select top n_seg
    ncut_evec = D_inv_sqrt.dot(vecs[:, :n_seg])

    # Transfer back to X
    # evec = Dx * B * Ncut_evec
    evec = Dx.dot(B).dot(ncut_evec)

    # Normalize rows
    row_norms = np.sqrt(np.sum(evec**2, axis=1)) + 1e-10
    evec = evec / row_norms[:, np.newaxis]

    # K-Means
    # Matlab: 'Replicates', 3. Scikit-learn: n_init=3.
    kmeans = KMeans(n_clusters=n_seg, init='random', n_init=3, max_iter=100)
    labels = kmeans.fit_predict(evec)

    return labels + 1
