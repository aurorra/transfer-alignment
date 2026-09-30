import numpy as np
from scipy.linalg import eigh


def _cov_shrink(X: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """
    Empirical covariance with diagonal shrinkage: Sigma + lam * I.
    X is (N,D). Rows are samples. Assumes X is already centered.
    """
    N = X.shape[0]
    if N <= 1:
        raise ValueError('Need at least 2 samples to estimate covariance.')
    Sigma = (X.T @ X) / (N - 1)
    # diagonal shrinkage for stability
    Sigma.flat[:: Sigma.shape[0] + 1] += lam
    return Sigma.astype(np.float32, copy=False)


def _sym_matrix_pow(mat: np.ndarray, p: float, eps: float = 1e-12) -> np.ndarray:
    """
    Symmetric matrix power using eigen-decomposition.
    mat must be SPD (we add shrinkage up-front).
    Returns Q * diag(vals**p) * Q^T with eigenvalues clipped to eps.
    """
    # eigh for symmetric matrices
    vals, vecs = eigh(mat)
    vals = np.clip(vals, eps, None).astype(np.float64, copy=False)
    Dp = np.diag(vals ** p)
    out = (vecs @ Dp) @ vecs.T
    return out.astype(np.float32, copy=False)


def fit_coral(X_src: np.ndarray, X_tgt: np.ndarray, lam: float = 1e-3) -> dict:
    """
    Fit CORAL whitening-coloring transform parameters.
    Returns dict with:
      - mu_src, mu_tgt: means
      - A: linear map = Sigma_tgt^{-1/2} Sigma_src^{1/2}
    """
    X_src = X_src.astype(np.float32, copy=False)
    X_tgt = X_tgt.astype(np.float32, copy=False)

    mu_src = X_src.mean(axis=0, keepdims=True).astype(np.float32)
    mu_tgt = X_tgt.mean(axis=0, keepdims=True).astype(np.float32)

    Xs = X_src - mu_src
    Xt = X_tgt - mu_tgt

    Sig_s = _cov_shrink(Xs, lam=lam)   # (D,D)
    Sig_t = _cov_shrink(Xt, lam=lam)   # (D,D)

    Sig_s_half   = _sym_matrix_pow(Sig_s, +0.5)
    Sig_t_mhalf  = _sym_matrix_pow(Sig_t, -0.5)

    A = Sig_t_mhalf @ Sig_s_half       # (D,D)
    return {'mu_src': mu_src, 'mu_tgt': mu_tgt, 'A': A.astype(np.float32)}


def coral_transform(X: np.ndarray, coral_params: dict) -> np.ndarray:
    """
    Apply CORAL transform learned from fit_coral to arbitrary X from target domain:
      X_coral = (X - mu_tgt) @ A + mu_src
    """
    X = X.astype(np.float32, copy=False)
    mu_src = coral_params['mu_src']
    mu_tgt = coral_params['mu_tgt']
    A = coral_params['A']
    return ((X - mu_tgt) @ A + mu_src).astype(np.float32, copy=False)