"""Initial estimators T: OLS and Huber M, both with an unpenalized intercept.

Huber follows the MASS::rlm method='M' IRLS loop (Venables and Ripley) with the
paper's settings: psi.huber k=1.345, scale.est='MAD' (center 0, divide by
0.6745), maxit=200, acc=1e-4, test.vec='resid'. This is not MASS::rlm MM.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from penalized_distillation.scad import (
    HUBER_ACC,
    HUBER_K,
    HUBER_MAD_DENOM,
    HUBER_MAXIT,
    MAD_CONSISTENCY,
    SIGMA_FLOOR,
)


@dataclass
class InitialFit:
    alpha: jnp.ndarray
    beta: jnp.ndarray
    fitted: jnp.ndarray
    residuals: jnp.ndarray
    sigma_hat: jnp.ndarray
    sigma_branch: str
    converged: bool
    n_iter: int
    name: str


def _as_xy(X, y):
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array of shape (n, p)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if X.shape[1] < 1:
        raise ValueError("X must have at least one column")
    return X, y


def design_with_intercept(X):
    n = X.shape[0]
    return jnp.concatenate([jnp.ones((n, 1), dtype=X.dtype), X], axis=1)


def residual_scale(residuals):
    """Operational residual scale (3).

    MAD is the median absolute deviation from the median, times 1.4826.
    Returns (sigma, branch) with branch in {'mad', 'rms', 'unit'}.
    """
    r = jnp.asarray(residuals, dtype=jnp.float64).reshape(-1)
    center = jnp.median(r)
    s_mad = MAD_CONSISTENCY * jnp.median(jnp.abs(r - center))
    s_rms = jnp.sqrt(jnp.mean(r * r))
    use_mad = jnp.isfinite(s_mad) & (s_mad > SIGMA_FLOOR)
    use_rms = jnp.isfinite(s_rms) & (s_rms > SIGMA_FLOOR)
    sigma = jnp.where(use_mad, s_mad, jnp.where(use_rms, s_rms, 1.0))
    if bool(use_mad):
        branch = "mad"
    elif bool(use_rms):
        branch = "rms"
    else:
        branch = "unit"
    return sigma, branch


def _lstsq(Z, y):
    coef, _, _, _ = jnp.linalg.lstsq(Z, y, rcond=None)
    return coef


def _wls(Z, y, weights):
    sw = jnp.sqrt(jnp.maximum(weights, 0.0))
    return _lstsq(Z * sw[:, None], y * sw)


def _pack(name, coef, y, Z, converged, n_iter):
    fitted = Z @ coef
    residuals = y - fitted
    sigma, branch = residual_scale(residuals)
    return InitialFit(
        alpha=coef[0],
        beta=coef[1:],
        fitted=fitted,
        residuals=residuals,
        sigma_hat=sigma,
        sigma_branch=branch,
        converged=converged,
        n_iter=n_iter,
        name=name,
    )


def ols(X, y):
    """Ordinary least squares with an unpenalized intercept."""
    X, y = _as_xy(X, y)
    Z = design_with_intercept(X)
    coef = _lstsq(Z, y)
    return _pack("ols", coef, y, Z, True, 1)


def huber_m(X, y, k=HUBER_K, maxit=HUBER_MAXIT, acc=HUBER_ACC):
    """Huber M-estimate by IRLS, MASS::rlm method='M' with psi.huber.

    Scale is re-estimated each iteration as median(|residual|) / 0.6745, which
    is MAD about 0, not about the median. Equation (3) is applied afterwards
    to the final residuals and is a separate scale.
    """
    X, y = _as_xy(X, y)
    Z = design_with_intercept(X)
    coef = _lstsq(Z, y)
    resid = y - Z @ coef
    converged = False
    n_iter = 0
    for n_iter in range(1, int(maxit) + 1):
        old = resid
        scale = jnp.median(jnp.abs(resid)) / HUBER_MAD_DENOM
        if float(scale) == 0.0:
            converged = True
            n_iter -= 1
            break
        weights = jnp.minimum(1.0, k / jnp.abs(resid / scale))
        coef = _wls(Z, y, weights)
        resid = y - Z @ coef
        numer = jnp.sum((old - resid) ** 2)
        denom = jnp.maximum(1e-20, jnp.sum(old ** 2))
        delta = jnp.sqrt(numer / denom)
        if float(delta) <= acc:
            converged = True
            break
    return _pack("huber", coef, y, Z, converged, n_iter)
