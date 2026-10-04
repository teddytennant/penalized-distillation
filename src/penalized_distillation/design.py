"""Seeded contaminated Gaussian design used by the demo and the recovery test."""

from __future__ import annotations

import numpy as np

BETA0 = np.array([3.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])


def ar1_covariance(p, rho=0.5):
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])


def contaminated_design(
    n=120,
    p=8,
    seed=0,
    rho=0.5,
    noise_sd=1.0,
    contamination=0.1,
    shift=8.0,
    beta0=None,
):
    """AR(1) covariates, Gaussian noise, and a fixed-size response shift.

    Contamination matches the paper's Monte Carlo: a uniformly sampled subset
    of responses, of size round(contamination * n), is shifted by `shift`.
    """
    if beta0 is None:
        beta0 = BETA0[:p] if p <= BETA0.shape[0] else np.pad(BETA0, (0, p - BETA0.shape[0]))
    beta0 = np.asarray(beta0, dtype=float).reshape(-1)
    if beta0.shape != (p,):
        raise ValueError("beta0 must have length p")
    rng = np.random.default_rng(seed)
    cov = ar1_covariance(p, rho)
    shocks = rng.normal(size=(n, p))
    X = shocks @ np.linalg.cholesky(cov).T
    y = X @ beta0 + rng.normal(scale=noise_sd, size=n)
    n_shift = int(round(contamination * n))
    if n_shift:
        rows = rng.choice(n, size=n_shift, replace=False)
        y = y.copy()
        y[rows] += shift
    return X, y, beta0
