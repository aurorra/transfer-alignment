import numpy as np
import ot


def compute_source_stats(X: np.ndarray, eps: float = 1e-12):
    mu = X.mean(axis=0)
    sigma = np.maximum(X.std(axis=0), eps)
    return mu, sigma


def apply_standardization(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma


def ot_map_target_to_source(
    Xs_train: np.ndarray,
    Xt_test: np.ndarray,
    reg: float = 0.05,  # regularization term >0 for Sinkhorn
    n_fit: int | None = None,   # take subset of samples if given a number
    standardize: bool = True,
    invert_standardization: bool = True,
    dtype=np.float32,
    numItermax: int = 200,  # max number of iterations for Sinkhorn
    seed: int = 0,
    tol: float = 1e-8,  # tolerance for convergence
    metric: str = 'sqeuclidean' # ground metric for the Wasserstein problem
):
    """
    Fits Sinkhorn on source/target embeddings, then barycentrically map target -> source space.
    Returns mapped embeddings for all target samples.
    """
    if reg <= 0 :
        raise ValueError('reg must be > 0 for Sinkhorn.')

    rng = np.random.default_rng(seed)

    Xs_full = np.asarray(Xs_train, dtype=dtype)
    Xt_full = np.asarray(Xt_test, dtype=dtype)

    # Choose supports for fitting
    if n_fit is None:
        Xs_fit = Xs_full
        Xt_fit = Xt_full
    else:
        n_fit = min(n_fit, len(Xs_full), len(Xt_full))
        idx_s = rng.choice(len(Xs_full), n_fit, replace=False)
        idx_t = rng.choice(len(Xt_full), n_fit, replace=False)
        Xs_fit = Xs_full[idx_s]
        Xt_fit = Xt_full[idx_t]

    # Standardize data if necessary
    stats = {}
    if standardize:
        # Compute stats once from raw source fit
        mu_s, sigma_s = compute_source_stats(Xs_fit)
        mu_t, sigma_t = compute_source_stats(Xt_fit)
        stats = {'mu_s': mu_s, 'sigma_s': sigma_s, 'mu_t': mu_t, 'sigma_t': sigma_t}

        # Apply to fit sets
        Xs_fit = apply_standardization(Xs_fit, mu_s, sigma_s).astype(dtype)
        Xt_fit = apply_standardization(Xt_fit, mu_t, sigma_t).astype(dtype)

        # Apply same stats to full target
        Xt_full = apply_standardization(Xt_full, mu_t, sigma_t).astype(dtype)

    # Sinkhorn Transport
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=reg, max_iter=numItermax, out_of_sample_map='ferradans', log=True,
                                          verbose=True, tol=tol, metric=metric)
    ot_sinkhorn.fit(Xs=Xt_fit, Xt=Xs_fit)

    # Transport target samples onto source samples
    transp_Xt_sinkhorn = ot_sinkhorn.transform(Xs=Xt_full)

    # Invert standardization (if done previously)
    if invert_standardization and standardize:
        transp_Xt_sinkhorn = transp_Xt_sinkhorn * stats['sigma_s'] + stats['mu_s']
    elif invert_standardization and (not standardize):
        print('Warning: invert_standardization=True but standardize=False. Nothing to invert.')

    return transp_Xt_sinkhorn
