import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import KernelPCA


# --- PCA alignment (linear) ---
def fit_pca(X: np.ndarray, n_components: int | str = 'all'):
    """
    Full-rank by default. Returns dict with mean, components (D x d), explained_variance (d,).
    Components are orthonormal (columns).
    """
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)  # compute mean
    Xc = X - mu     # center data before SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # run SVD on centered data
    if n_components == 'all':
        n_components = Vt.shape[0]
    V = Vt[:n_components].T           # D x d  (columns are PCs)
    ev = (S[:n_components]**2) / (X.shape[0] - 1)   # explained variance
    return {'mean': mu.astype(np.float32), 'components': V.astype(np.float32), 'explained_variance': ev.astype(np.float32)}


def pca_transform(X: np.ndarray, pca: dict):
    Xc = X.astype(np.float32, copy=False) - pca['mean'] # center using mean from training set
    return Xc @ pca['components']   # project into PCA coordinates (N x d)


def pca_inverse(Z: np.ndarray, pca: dict):
    return Z @ pca['components'].T + pca['mean']


def orthogonal_align_bases(pca_src: dict, pca_tgt: dict, k: int | str = 'all', allow_scale: bool = False):
    Vs = pca_src['components']  # D x d_s
    Vt = pca_tgt['components']  # D x d_t
    d_s, d_t = Vs.shape[1], Vt.shape[1]
    d = min(d_s, d_t) if (k == 'all' or k is None) else int(k)

    A = Vs[:, :d]                          # D x d  (src top-d)
    B = Vt[:, :d]                          # D x d  (tgt top-d)
    R, scale = orthogonal_procrustes(B, A) # B @ R ≈ A  -> R is (d x d)
    if not allow_scale:
        scale = 1.0
    return R.astype(np.float32), float(scale)


def pca_align_and_backproject(
    X_src: np.ndarray,
    pca_src: dict,  # type 1 pca
    pca_tgt: dict,  # type 2 pca
    *,
    k: int | str = 'all',
    allow_scale: bool = False,
    pca_space: bool = True,
) -> np.ndarray:

    # 1) Project into PCA space
    if pca_space:
        Z_t = pca_transform(X_src, pca_tgt)              # N x d_t
    else:
        Z_t = X_src

    d_s = pca_src['components'].shape[1]
    d_t = pca_tgt['components'].shape[1]
    d = min(d_s, d_t) if (k == 'all' or k is None) else int(k)

    # 2) Rotate only the top-d coords using Procrustes closed form solution
    Z_top = Z_t[:, :d]                               # N x d
    R, c = orthogonal_align_bases(pca_src, pca_tgt, k=d, allow_scale=allow_scale)  # d x d
    Z_aligned_top = (Z_top @ R) * c                  # N x d

    # 3) pad (if needed) to the src PCA width, then inverse
    if d_s > d:
        Z_full = np.zeros((Z_aligned_top.shape[0], d_s), dtype=np.float32)
        Z_full[:, :d] = Z_aligned_top
    else:
        Z_full = Z_aligned_top[:, :d_s]

    if pca_space:
        X_back = pca_inverse(Z_full, pca_src)            # N x D
    else:
        X_back = Z_full

    return X_back.astype(np.float32, copy=False)


# --- Kernel PCA alignment with RBF kernel (nonlinear) ---
def _topk_basis(Z, k):
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt.T[:, :k]  # (d x k)


def _stratified_idx(y, n, seed=0):
    y = np.asarray(y).astype(int)
    N = len(y)
    if n is None or n >= N:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return rng.choice(N, size=n, replace=False)
    n0 = max(1, min(len(idx0), int(round(n * len(idx0) / N))))
    n1 = max(1, min(len(idx1), n - n0))
    # fix rounding
    while n0 + n1 < n and n0 < len(idx0): n0 += 1
    while n0 + n1 < n and n1 < len(idx1): n1 += 1
    while n0 + n1 > n and n0 > 1: n0 -= 1
    sub = np.concatenate([
        rng.choice(idx0, size=n0, replace=False),
        rng.choice(idx1, size=n1, replace=False)
    ])
    rng.shuffle(sub)
    return sub


def _stratified_idx_min_pos(y, n, seed=0, min_pos_frac=0.25):
    """
    Source-side sampling for imbalanced data.

    Ensures that at least min_pos_frac of the sampled source support set
    consists of positive/error examples, if enough positives are available.

    This uses source labels only. Do not use target labels for alignment.
    """
    y = np.asarray(y).astype(int)
    N = len(y)

    if n is None or n >= N:
        return np.arange(N)

    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        return rng.choice(N, size=n, replace=False)

    n = int(n)

    n1 = int(round(n * float(min_pos_frac)))
    n1 = max(1, n1)
    n1 = min(n1, len(idx1), n - 1)

    n0 = n - n1
    n0 = min(n0, len(idx0))

    # If one class was too small, fill the remaining samples from the other class
    while n0 + n1 < n and n1 < len(idx1):
        n1 += 1
    while n0 + n1 < n and n0 < len(idx0):
        n0 += 1

    sub = np.concatenate([
        rng.choice(idx0, size=n0, replace=False),
        rng.choice(idx1, size=n1, replace=False),
    ])

    rng.shuffle(sub)
    return sub


def _batched_apply(func, X, batch_size=None):
    """
    Apply KPCA transform/inverse_transform in batches.

    Useful for large target sets, e.g. 100,000 samples.
    """
    if batch_size is None:
        return func(X)

    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size >= len(X):
        return func(X)

    chunks = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        chunks.append(func(X[start:end]))

    return np.vstack(chunks)


def kpca_align_and_preimage(
        X1_train,
        y1_train,
        X2_test,
        *,
        X2_alignment=None,
        n_fit=8000,
        n_components=128,
        k_align=64,
        gamma=1,
        alpha=1e-2,
        seed=0,
        sampling='stratified',
        min_pos_frac=0.25,
        transform_batch_size=None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """
    Nonlinear (RBF-KPCA) analogue of PCA alignment that returns aligned embeddings in R^D.

    Uses:
      - stratified subset on source (needs y1_train)
      - random subset on target (no target labels)
      - aligns dominant directions (bases) in KPCA coordinate space via Procrustes
      - uses KernelPCA.inverse_transform as approximate preimage (regularized by alpha)

    Returns:
      X2_aligned: (N2, D) float32
      meta: dict with chosen hyperparams
    """
    X1_train = np.asarray(X1_train, dtype=np.float32)
    X2_test  = np.asarray(X2_test,  dtype=np.float32)
    if X2_alignment is None:
        X2_alignment = X2_test
    else:
        X2_alignment = np.asarray(X2_alignment, dtype=np.float32)

    # --- 1) source subset ---
    if sampling == 'stratified':
        i1 = _stratified_idx(y1_train, n_fit, seed=seed)
    elif sampling == 'min_pos':
        i1 = _stratified_idx_min_pos(
            y1_train,
            n_fit,
            seed=seed,
            min_pos_frac=min_pos_frac,
        )
    else:
        raise ValueError(f'Unknown KPCA sampling strategy: {sampling}')

    X1_fit = X1_train[i1]
    y1_fit = np.asarray(y1_train).astype(int)[i1]

    # --- 2) target subset (random, no labels needed) ---
    rng = np.random.default_rng(seed + 1)
    n2 = X2_alignment.shape[0]
    n_fit_t = min(int(n_fit), n2)
    i2 = rng.choice(n2, size=n_fit_t, replace=False)
    X2_fit = X2_alignment[i2]

    # --- 3) component bounds ---
    # KernelPCA requires n_components <= n_samples_fit - 1
    n_components = int(min(n_components, len(X1_fit) - 1))
    n_components = max(2, n_components)
    k_align = int(min(k_align, n_components))
    k_align = max(1, k_align)

    # --- 4) fit KPCA on source subset, enable inverse for preimage ---
    kpca = KernelPCA(
        n_components=n_components,
        kernel='rbf',
        gamma=float(gamma),
        fit_inverse_transform=True,
        alpha=float(alpha),     # inverse-transform regularization
        eigen_solver='auto',
        random_state=seed,
    )

    Z1_fit = kpca.fit_transform(X1_fit)    # (n_fit_src, d)
    Z2_fit = kpca.transform(X2_fit)        # (n_fit_tgt, d)

    # --- 5) align dominant directions (bases) in KPCA coord space ---
    B1 = _topk_basis(Z1_fit, k_align)      # (d, k)
    B2 = _topk_basis(Z2_fit, k_align)      # (d, k)
    R, _ = orthogonal_procrustes(B2, B1)   # (k, k), B2 @ R ≈ B1

    # --- 6) transform full target, rotate top-k coords, then preimage ---
    if transform_batch_size:
        Z2 = _batched_apply(
            kpca.transform,
            X2_test,
            batch_size=transform_batch_size,
        )
    else:
        Z2 = kpca.transform(X2_test)           # (N2, d)

    mu1 = Z1_fit.mean(0, keepdims=True)
    mu2 = Z2_fit.mean(0, keepdims=True)
    Z2c = Z2 - mu2

    # coordinates in target subspace (N2, k)
    A = Z2c @ B2
    # rotate within subspace (N2, k)
    A_rot = A @ R.T  # because B2 @ R ≈ B1
    # reconstruct aligned subspace component in source basis (N2, d)
    Z_par = A_rot @ B1.T
    # preserve orthogonal complement
    Z_perp = Z2c - (A @ B2.T)
    Z2a = Z_par + Z_perp
    Z2a = Z2a + mu1
    if transform_batch_size:
        X2_aligned = _batched_apply(
            kpca.inverse_transform,
            Z2a,
            batch_size=transform_batch_size,
        ).astype(np.float32, copy=False)
    else:
        X2_aligned = kpca.inverse_transform(Z2a).astype(np.float32, copy=False)

    if not np.allclose(B1.T @ B1, np.eye(k_align), atol=1e-3):
        print('[WARNING] np.allclose(B1.T @ B1, np.eye(k_align), atol=1e-3) resulted in False')
    if not np.allclose(B2.T @ B2, np.eye(k_align), atol=1e-3):
        print('[WARNING] np.allclose(B2.T @ B2, np.eye(k_align), atol=1e-3) resulted in False')
    if np.max(np.abs(np.linalg.norm(A, axis=1) - np.linalg.norm(A_rot, axis=1))) > 1e-3:
        print('[WARNING] np.max(np.abs(np.linalg.norm(A, axis=1) - np.linalg.norm(A_rot, axis=1))) > 1e-3')
    if np.max(np.abs(Z2c - ((A @ B2.T) + Z_perp))) > 1e-3:
        print('[WARNING] decomposition check failed')
    err = np.linalg.norm(B2 @ R - B1, ord='fro') / np.linalg.norm(B1, ord='fro')
    if err > 1e-2:
        print('[WARNING] Procrustes fit is poor:', err)

    meta = dict(
        n_fit_src=int(len(X1_fit)),
        n_fit_tgt=int(len(X2_fit)),
        n_components=int(n_components),
        k_align=int(k_align),
        gamma=float(gamma),
        alpha=float(alpha),
        seed=int(seed),
        sampling=str(sampling),
        min_pos_frac=float(min_pos_frac),
        n_pos_src_fit=int(np.sum(y1_fit == 1)),
        n_neg_src_fit=int(np.sum(y1_fit == 0)),
        pos_frac_src_fit=float(np.mean(y1_fit == 1)),
        transform_batch_size=None if transform_batch_size is None else int(transform_batch_size),
    )
    return X2_aligned, meta