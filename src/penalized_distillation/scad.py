"""SCAD penalty, orthogonal closed form, and the scalar coordinate update.

Equation numbers refer to Shin and Park, arXiv:2609.23937.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Fan and Li (2001), Table 27.
SCAD_A = 3.7

# Numerical thresholds from Table 27 and equation (3).
DF_THRESHOLD = 1e-10
SUPPORT_THRESHOLD = 1e-8
SIGMA_FLOOR = 1e-10
GIC_SIGMA_FLOOR = 1e-12
MAD_CONSISTENCY = 1.4826

# Path defaults, Table 27 and Appendix D.2.
N_LAMBDA = 100
LAMBDA_MIN_RATIO = 1e-3
LAMBDA_MIN_RATIO_HIGH_P = 0.05
MAX_SWEEPS = 10_000
TOL_FACTOR = 1e-5
LAMBDA_MAX_INFLATION = 1e-4
LAMBDA_MAX_FALLBACK = 1e-6
KAPPA = 2.0

# MASS::rlm Huber defaults used by the paper (Table 27: 200 iterations).
HUBER_K = 1.345
HUBER_MAXIT = 200
HUBER_ACC = 1e-4
HUBER_MAD_DENOM = 0.6745


def scad_penalty(theta, lam, a=SCAD_A):
    """Fan and Li SCAD penalty p_lam(|theta|) with constant a.

    p(t) = lam * t                         if t <= lam
         = -(t^2 - 2 a lam t + lam^2) / (2 (a-1))   if lam < t <= a lam
         = (a+1) lam^2 / 2                 if t > a lam
    with t = |theta|. Continuous at the knots. Zero when lam <= 0.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    t = jnp.abs(theta)
    a1 = a - 1.0
    p_linear = lam * t
    p_quad = -(t * t - 2.0 * a * lam * t + lam * lam) / (2.0 * a1)
    p_flat = (a + 1.0) * lam * lam / 2.0
    pen = jnp.where(t <= lam, p_linear, jnp.where(t <= a * lam, p_quad, p_flat))
    return jnp.where(lam <= 0.0, jnp.zeros_like(pen), pen)


def theta_scad(z, lam, a=SCAD_A):
    """Orthogonal closed form Theta_SCAD(z; lambda), equation (5).

    Valid for the profiled slope problem when X^T X / n = I.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    az = jnp.abs(z)
    sign = jnp.sign(z)
    soft = sign * jnp.maximum(az - lam, 0.0)
    middle = ((a - 1.0) * z - a * lam * sign) / (a - 2.0)
    return jnp.where(az <= 2.0 * lam, soft, jnp.where(az <= a * lam, middle, z))


def scalar_objective(b, zeta, v, ell, a=SCAD_A):
    """One-coordinate conditional objective q_j, equation (6), up to a constant."""
    b = jnp.asarray(b, dtype=jnp.float64)
    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    ell = jnp.asarray(ell, dtype=jnp.float64)
    return 0.5 * v * b * b - zeta * b + scad_penalty(b, ell, a)


def _in_open_ray(value, sign, lo, hi):
    """True when value lies in sign * (lo, hi), Appendix D.1."""
    magnitude = sign * value
    return (sign != 0.0) & (magnitude > lo) & (magnitude < hi)


def scalar_update(b, zeta, v, ell, a=SCAD_A):
    """Global minimizer of q_j. On an exact floating-point tie, keep b.

    Appendix D.1. Unpenalized coordinates (ell <= 0), including the intercept,
    use the least-squares update zeta / v. A zero column (v <= 0) is set to 0
    when penalized and left unchanged when not.
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    ell = jnp.asarray(ell, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)

    u = jnp.abs(zeta)
    sign = jnp.sign(zeta)
    a1 = a - 1.0
    safe_v = jnp.where(v > 0.0, v, 1.0)
    stat_soft = sign * (u - ell) / safe_v
    denom = v - 1.0 / a1
    safe_denom = jnp.where(denom != 0.0, denom, 1.0)
    stat_middle = sign * (u - a * ell / a1) / safe_denom
    stat_flat = zeta / safe_v

    zero = jnp.zeros_like(b)
    cands = jnp.stack(
        [b, zero, sign * ell, sign * a * ell, stat_soft, stat_middle, stat_flat]
    )
    valid = jnp.stack(
        [
            jnp.ones_like(b, dtype=jnp.bool_),
            jnp.ones_like(b, dtype=jnp.bool_),
            jnp.ones_like(b, dtype=jnp.bool_),
            jnp.ones_like(b, dtype=jnp.bool_),
            _in_open_ray(stat_soft, sign, 0.0, ell),
            (v > 1.0 / a1) & _in_open_ray(stat_middle, sign, ell, a * ell),
            _in_open_ray(stat_flat, sign, a * ell, jnp.inf),
        ]
    )
    qs = jnp.stack(
        [0.5 * v * cands[i] ** 2 - zeta * cands[i] + scad_penalty(cands[i], ell, a) for i in range(7)]
    )
    qs = jnp.where(valid, qs, jnp.inf)
    # argmin returns the first minimum, and the current coefficient is first,
    # so an exact tie keeps the current value.
    chosen = cands[jnp.argmin(qs)]

    unpenalized = ell <= 0.0
    no_curvature = v <= 0.0
    chosen = jnp.where(unpenalized & ~no_curvature, zeta / safe_v, chosen)
    chosen = jnp.where(~unpenalized & no_curvature, zero, chosen)
    chosen = jnp.where(unpenalized & no_curvature, b, chosen)
    return chosen


def full_objective(b, X, y, lam, penalty_factors, a=SCAD_A):
    """Explicit-intercept SCAD objective (4), recomputed from X and y.

    b is (intercept, slopes). The intercept is never penalized. Slope j is
    penalized by p_{lam * f_j}.
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    f = jnp.asarray(penalty_factors, dtype=jnp.float64)
    n = y.shape[0]
    resid = y - b[0] - X @ b[1:]
    quad = jnp.sum(resid * resid) / (2.0 * n)
    ell = lam * f[1:]
    pen = jnp.sum(jax.vmap(lambda bj, ej: scad_penalty(bj, ej, a))(b[1:], ell))
    return quad + pen


def residual_sum_of_squares(b, X, y):
    """||y - alpha 1 - X beta||^2. Not divided by n."""
    resid = y - b[0] - X @ b[1:]
    return jnp.sum(resid * resid)


def degrees_of_freedom(beta, threshold=DF_THRESHOLD):
    """Intercept plus slopes with |beta_j| > threshold (Table 27: 1e-10)."""
    beta = jnp.asarray(beta, dtype=jnp.float64)
    return 1 + jnp.sum(jnp.abs(beta) > threshold)


def support_indices(beta, threshold=SUPPORT_THRESHOLD):
    """Slope indices with |beta_j| > threshold (Table 27: 1e-8)."""
    beta = jnp.asarray(beta, dtype=jnp.float64)
    return jnp.where(jnp.abs(beta) > threshold)[0]
