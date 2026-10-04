"""Algorithm 1 and the raw-response SCAD comparator."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from penalized_distillation.initial import huber_m, ols
from penalized_distillation.path import scad_path
from penalized_distillation.scad import (
    DF_THRESHOLD,
    GIC_SIGMA_FLOOR,
    KAPPA,
    SUPPORT_THRESHOLD,
    degrees_of_freedom,
    residual_sum_of_squares,
    support_indices,
)

_ESTIMATORS = {
    "ols": ols,
    "huber": huber_m,
}


@dataclass
class DistillResult:
    estimator: str
    initial: object
    alpha_tilde: jnp.ndarray
    beta_tilde: jnp.ndarray
    sigma_hat: jnp.ndarray
    sigma_branch: str
    y_tilde: jnp.ndarray
    path: object
    rss_d: jnp.ndarray
    df: jnp.ndarray
    gic: jnp.ndarray
    index_hat: int
    lambda_hat: jnp.ndarray
    alpha_hat: jnp.ndarray
    beta_hat: jnp.ndarray
    support: jnp.ndarray
    kappa: float


@dataclass
class RawResult:
    path: object
    rss: jnp.ndarray
    df: jnp.ndarray
    bic: jnp.ndarray
    index_hat: int
    lambda_hat: jnp.ndarray
    alpha_hat: jnp.ndarray
    beta_hat: jnp.ndarray
    support: jnp.ndarray


def fidelity_gic(rss, sigma, n, df, kappa=KAPPA):
    """Algorithm 1, step 3.

    RSS_d(lambda) / max(sigma_hat^2, 1e-12) + kappa * log(n) * df(lambda).
    df includes the intercept.
    """
    rss = jnp.asarray(rss, dtype=jnp.float64)
    df = jnp.asarray(df, dtype=jnp.float64)
    scale = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64) ** 2, GIC_SIGMA_FLOOR)
    return rss / scale + kappa * jnp.log(n) * df


def raw_bic(rss, n, df):
    """Appendix D.2 raw-response selector: n log(RSS/n + 1e-12) + log(n) df."""
    rss = jnp.asarray(rss, dtype=jnp.float64)
    df = jnp.asarray(df, dtype=jnp.float64)
    return n * jnp.log(rss / n + 1e-12) + jnp.log(n) * df


def _path_rss_df(path, X, response):
    rss = jax.vmap(lambda b: residual_sum_of_squares(b, X, response))(path.coefficients)
    df = jax.vmap(lambda beta: degrees_of_freedom(beta, DF_THRESHOLD))(path.beta)
    return rss, df


def distill(X, y, estimator="huber", *, kappa=KAPPA, **path_kwargs):
    """Penalized distillation, Algorithm 1.

    estimator is 'huber', 'ols', or a callable returning an InitialFit. The
    second-stage path is fit to y_tilde = alpha_tilde + X beta_tilde, never to
    y. Ties in the GIC are broken toward the largest lambda (first grid point
    on the decreasing path).
    """
    if isinstance(estimator, str):
        if estimator not in _ESTIMATORS:
            raise ValueError("estimator must be 'ols', 'huber', or a callable")
        name = estimator
        fit = _ESTIMATORS[estimator](X, y)
    else:
        name = getattr(estimator, "__name__", "custom")
        fit = estimator(X, y)
    path = scad_path(X, fit.fitted, **path_kwargs)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    rss, df = _path_rss_df(path, X_arr, fit.fitted)
    gic = fidelity_gic(rss, fit.sigma_hat, path.n, df, kappa=kappa)
    index = int(jnp.argmin(gic))
    beta_hat = path.beta[index]
    return DistillResult(
        estimator=name,
        initial=fit,
        alpha_tilde=fit.alpha,
        beta_tilde=fit.beta,
        sigma_hat=fit.sigma_hat,
        sigma_branch=fit.sigma_branch,
        y_tilde=fit.fitted,
        path=path,
        rss_d=rss,
        df=df,
        gic=gic,
        index_hat=index,
        lambda_hat=path.lambdas[index],
        alpha_hat=path.alpha[index],
        beta_hat=beta_hat,
        support=support_indices(beta_hat, SUPPORT_THRESHOLD),
        kappa=float(kappa),
    )


def scad_raw(X, y, **path_kwargs):
    """Raw-response SCAD: the same solver, with stage-2 response equal to y.

    Selection uses the paper's raw criterion, not the distillation GIC.
    """
    path = scad_path(X, y, **path_kwargs)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    rss, df = _path_rss_df(path, X_arr, y_arr)
    bic = raw_bic(rss, path.n, df)
    index = int(jnp.argmin(bic))
    beta_hat = path.beta[index]
    return RawResult(
        path=path,
        rss=rss,
        df=df,
        bic=bic,
        index_hat=index,
        lambda_hat=path.lambdas[index],
        alpha_hat=path.alpha[index],
        beta_hat=beta_hat,
        support=support_indices(beta_hat, SUPPORT_THRESHOLD),
    )
