"""Safeguarded SCAD coordinate-descent path (Algorithm 1, step 2; Appendix D)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from penalized_distillation.initial import _as_xy, design_with_intercept
from penalized_distillation.scad import (
    LAMBDA_MAX_FALLBACK,
    LAMBDA_MAX_INFLATION,
    LAMBDA_MIN_RATIO,
    LAMBDA_MIN_RATIO_HIGH_P,
    MAX_SWEEPS,
    N_LAMBDA,
    SCAD_A,
    SIGMA_FLOOR,
    TOL_FACTOR,
    full_objective,
    scalar_update,
)


@dataclass
class PathResult:
    lambdas: jnp.ndarray
    coefficients: jnp.ndarray
    iterations: jnp.ndarray
    fixed_point_gap: jnp.ndarray
    objective_start: jnp.ndarray
    objective_candidate: jnp.ndarray
    objective_retained: jnp.ndarray
    accepted: jnp.ndarray
    failure_reason: tuple[str, ...]
    lambda_max: jnp.ndarray
    n: int
    p: int

    @property
    def alpha(self):
        return self.coefficients[:, 0]

    @property
    def beta(self):
        return self.coefficients[:, 1:]


def penalty_factor_vector(p, penalty_factors=None):
    """f_0 = 0 and f_j = 1 for slopes, unless a slope factor vector is supplied."""
    if penalty_factors is None:
        slopes = jnp.ones(p, dtype=jnp.float64)
    else:
        slopes = jnp.asarray(penalty_factors, dtype=jnp.float64).reshape(-1)
        if slopes.shape != (p,):
            raise ValueError("penalty_factors must have one entry per slope")
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), slopes])


def null_lambda_max(X, y, a=SCAD_A, penalty_factors=None):
    """Appendix D.2 null-grid upper endpoint.

    The path starts at the intercept-only least-squares state. If r0 is its
    residual and v_j = ||Z_j||^2 / n,

        lambda_max = (1 + 1e-4) * max_{j: f_j > 0}
            (|Z_j^T r0| / n) / (f_j * min{1, sqrt((a+1) v_j)}).

    If that maximum score is exactly 0, lambda_max = 1e-6.
    """
    X, y = _as_xy(X, y)
    n, p = X.shape
    f = penalty_factor_vector(p, penalty_factors)
    alpha0 = jnp.mean(y)
    r0 = y - alpha0
    v = jnp.sum(X * X, axis=0) / n
    score_num = jnp.abs(X.T @ r0) / n
    denom = f[1:] * jnp.minimum(1.0, jnp.sqrt((a + 1.0) * jnp.maximum(v, 0.0)))
    raw = jnp.where((f[1:] > 0.0) & (denom > 0.0), score_num / denom, 0.0)
    raw = jnp.where(jnp.isfinite(raw), raw, 0.0)
    max_score = jnp.max(raw)
    inflated = (1.0 + LAMBDA_MAX_INFLATION) * max_score
    return jnp.where(max_score == 0.0, LAMBDA_MAX_FALLBACK, inflated)


def lambda_grid(lam_max, n, p, n_lambda=N_LAMBDA, min_ratio=None):
    """Decreasing geometric grid from lam_max down to min_ratio * lam_max.

    Default min_ratio is 1e-3 when n > p + 1, else 0.05. Both endpoints are
    included. A one-point grid is [lam_max].
    """
    if n_lambda < 1:
        raise ValueError("n_lambda must be at least 1")
    if min_ratio is None:
        min_ratio = LAMBDA_MIN_RATIO if n > p + 1 else LAMBDA_MIN_RATIO_HIGH_P
    lam_max = jnp.asarray(lam_max, dtype=jnp.float64)
    if n_lambda == 1:
        return lam_max.reshape(1)
    exponents = jnp.linspace(0.0, 1.0, int(n_lambda))
    return lam_max * (jnp.asarray(min_ratio, dtype=jnp.float64) ** exponents)


def gram_system(X, y):
    """Z = [1, X], G = Z^T Z / n, c = Z^T y / n. Section 2.3."""
    Z = design_with_intercept(X)
    n = y.shape[0]
    G = (Z.T @ Z) / n
    c = (Z.T @ y) / n
    return Z, G, c


def intercept_only_state(y, p):
    """Least-squares intercept with every slope held at 0."""
    b = jnp.zeros(p + 1, dtype=jnp.float64)
    return b.at[0].set(jnp.mean(y))


def response_rms(y):
    """RMS of the stage-2 response. Unit fallback when it is nonfinite or ~0."""
    rms = jnp.sqrt(jnp.mean(jnp.asarray(y, dtype=jnp.float64) ** 2))
    return jnp.where(jnp.isfinite(rms) & (rms > SIGMA_FLOOR), rms, 1.0)


def convergence_tolerance(y, tol_factor=TOL_FACTOR):
    """1e-5 * RMS(y), with a unit fallback when RMS is nonfinite or ~0."""
    return float(tol_factor) * response_rms(y)


def _sweep(b, G, c, lam, f, a):
    def body(j, b):
        gb_j = jnp.dot(G[j], b)
        v = G[j, j]
        zeta = c[j] - gb_j + v * b[j]
        new = scalar_update(b[j], zeta, v, lam * f[j], a)
        return b.at[j].set(new)

    return jax.lax.fori_loop(0, b.shape[0], body, b)


def _targets(b, G, c, lam, f, a):
    """Simultaneous scalar updates T_lambda,j(b), other coordinates held fixed."""
    gb = G @ b
    v = jnp.diag(G)
    zeta = c - gb + v * b
    ell = lam * f

    def one(bj, zj, vj, ej):
        return scalar_update(bj, zj, vj, ej, a)

    return jax.vmap(one)(b, zeta, v, ell)


def _gap(b, G, c, lam, f, a, scale):
    targets = _targets(b, G, c, lam, f, a)
    return jnp.max(scale * jnp.abs(targets - b))


@jax.jit
def solve_at_lambda(b0, G, c, lam, f, a, tol, max_sweeps):
    """Warm-started Gauss-Seidel sweeps until both convergence checks pass.

    A sweep is accepted as converged only when the largest column-norm-scaled
    coordinate change and the recomputed fixed-point gap

        Delta_lambda(b) = max_j sqrt(v_j) |T_lambda,j(b) - b_j|

    are both strictly below tol. sqrt(v_j) = ||Z_j|| / sqrt(n) is the
    column-norm scaling used for both checks (Appendix D.2).
    """
    max_sweeps = jnp.asarray(max_sweeps, dtype=jnp.int32)
    scale = jnp.sqrt(jnp.maximum(jnp.diag(G), 0.0))

    def cond(state):
        _b, it, _change, _gap_value, done = state
        return (it < max_sweeps) & (~done)

    def body(state):
        b, it, _change_value, _gap_value, _done = state
        b_new = _sweep(b, G, c, lam, f, a)
        change = jnp.max(scale * jnp.abs(b_new - b))
        gap = _gap(b_new, G, c, lam, f, a, scale)
        done = (change < tol) & (gap < tol)
        return b_new, it + 1, change, gap, done

    init = (
        b0,
        jnp.int32(0),
        jnp.array(jnp.inf, dtype=jnp.float64),
        jnp.array(jnp.inf, dtype=jnp.float64),
        jnp.bool_(False),
    )
    b, it, change, gap, done = jax.lax.while_loop(cond, body, init)
    gap = jnp.where(it == 0, _gap(b0, G, c, lam, f, a, scale), gap)
    change = jnp.where(it == 0, 0.0, change)
    return b, it, gap, done, change


def accept_candidate(objective_start, objective_candidate, finite, passed_fixed_point):
    """Appendix D.2 safeguard.

    Retain a candidate only when it is finite, passed the fixed-point test, and
    its objective does not exceed the warm-start objective at the same lambda.
    The first failing condition is the recorded reason.
    """
    if not finite:
        return False, "nonfinite"
    if not passed_fixed_point:
        return False, "fixed_point"
    if objective_candidate > objective_start:
        return False, "objective_increase"
    return True, ""


def scad_path(
    X,
    response,
    lambdas=None,
    *,
    a=SCAD_A,
    n_lambda=N_LAMBDA,
    min_ratio=None,
    max_sweeps=MAX_SWEEPS,
    tol_factor=TOL_FACTOR,
    penalty_factors=None,
):
    """Safeguarded SCAD path for a stage-2 response.

    The response is y_tilde in distillation and the raw y in SCAD-raw. The
    path starts at the intercept-only least squares state and warm-starts each
    later grid point from the retained state. Max sweeps defaults to 10000 and
    is exposed so a caller can use fewer.
    """
    X, response = _as_xy(X, response)
    n, p = X.shape
    f = penalty_factor_vector(p, penalty_factors)
    _Z, G, c = gram_system(X, response)
    lam_max = null_lambda_max(X, response, a=a, penalty_factors=penalty_factors)
    if lambdas is None:
        grid = lambda_grid(lam_max, n, p, n_lambda=n_lambda, min_ratio=min_ratio)
    else:
        grid = jnp.asarray(lambdas, dtype=jnp.float64).reshape(-1)
        if grid.shape[0] < 1:
            raise ValueError("lambdas must contain at least one value")
    tol = convergence_tolerance(response, tol_factor=tol_factor)
    b = intercept_only_state(response, p)

    coefs = []
    iterations = []
    gaps = []
    obj_start = []
    obj_cand = []
    obj_ret = []
    accepted = []
    reasons = []

    for lam in grid:
        start = full_objective(b, X, response, lam, f, a)
        cand, it, gap, passed, _change = solve_at_lambda(
            b, G, c, lam, f, a, tol, int(max_sweeps)
        )
        cand_obj = full_objective(cand, X, response, lam, f, a)
        finite = bool(jnp.isfinite(cand_obj) and jnp.all(jnp.isfinite(cand)))
        ok, reason = accept_candidate(float(start), float(cand_obj), finite, bool(passed))
        if ok:
            retained = cand
            retained_obj = cand_obj
            b = cand
        else:
            retained = b
            retained_obj = start
        coefs.append(retained)
        iterations.append(it)
        gaps.append(gap)
        obj_start.append(start)
        obj_cand.append(cand_obj)
        obj_ret.append(retained_obj)
        accepted.append(ok)
        reasons.append(reason)

    return PathResult(
        lambdas=grid,
        coefficients=jnp.stack(coefs),
        iterations=jnp.stack(iterations),
        fixed_point_gap=jnp.stack(gaps),
        objective_start=jnp.stack(obj_start),
        objective_candidate=jnp.stack(obj_cand),
        objective_retained=jnp.stack(obj_ret),
        accepted=jnp.array(accepted),
        failure_reason=tuple(reasons),
        lambda_max=lam_max,
        n=n,
        p=p,
    )
