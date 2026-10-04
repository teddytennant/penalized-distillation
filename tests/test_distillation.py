"""Equation checks for Shin and Park, arXiv:2609.23937, Algorithm 1."""

import numpy as np
import jax.numpy as jnp
import pytest

from penalized_distillation import distill, huber_m, ols, scad_path
from penalized_distillation.design import contaminated_design
from penalized_distillation.distill import fidelity_gic
from penalized_distillation.initial import InitialFit, residual_scale
from penalized_distillation.path import (
    accept_candidate,
    gram_system,
    intercept_only_state,
    null_lambda_max,
    penalty_factor_vector,
)
from penalized_distillation.scad import (
    SCAD_A,
    degrees_of_freedom,
    full_objective,
    scalar_update,
    theta_scad,
)


A = SCAD_A


def _closed_form(z, lam, a=A):
    """Theta_SCAD, independent of the library implementation."""
    z = np.asarray(z, dtype=float)
    out = np.empty(np.shape(z), dtype=float)
    flat = out.ravel()
    for i, zi in enumerate(np.asarray(z, dtype=float).ravel()):
        az = abs(zi)
        sign = np.sign(zi)
        if az <= 2.0 * lam:
            flat[i] = sign * max(az - lam, 0.0)
        elif az <= a * lam:
            flat[i] = ((a - 1.0) * zi - a * lam * sign) / (a - 2.0)
        else:
            flat[i] = zi
    return out


def _penalty(theta, lam, a=A):
    t = abs(float(theta))
    lam = float(lam)
    if lam <= 0.0:
        return 0.0
    if t <= lam:
        return lam * t
    if t <= a * lam:
        return -(t * t - 2.0 * a * lam * t + lam * lam) / (2.0 * (a - 1.0))
    return (a + 1.0) * lam * lam / 2.0


def _q(b, zeta, v, ell, a=A):
    return 0.5 * v * b * b - zeta * b + _penalty(b, ell, a)


def _feasible_stationary(zeta, v, ell, a=A):
    """Appendix D.1 stationary points that lie in their open pieces."""
    if ell <= 0.0 or v <= 0.0 or zeta == 0.0:
        return []
    u = abs(zeta)
    sign = np.sign(zeta)
    points = []
    soft = sign * (u - ell) / v
    if 0.0 < sign * soft < ell:
        points.append(soft)
    if v > 1.0 / (a - 1.0):
        middle = sign * (u - a * ell / (a - 1.0)) / (v - 1.0 / (a - 1.0))
        if ell < sign * middle < a * ell:
            points.append(middle)
    flat = zeta / v
    if sign * flat > a * ell:
        points.append(flat)
    return points


def _centered_orthogonal(n, p, seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, p))
    raw -= raw.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(raw, mode="reduced")
    q -= q.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(q, mode="reduced")
    return q * np.sqrt(n)


def _fixed_estimator(alpha, beta):
    alpha = float(alpha)
    beta = np.asarray(beta, dtype=float)

    def estimator(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        fitted = alpha + X @ beta
        return InitialFit(
            alpha=jnp.asarray(alpha),
            beta=jnp.asarray(beta),
            fitted=jnp.asarray(fitted),
            residuals=jnp.asarray(y - fitted),
            sigma_hat=jnp.asarray(1.0),
            sigma_branch="unit",
            converged=True,
            n_iter=0,
            name="fixed",
        )

    return estimator


def _lambda_max_formula(X, y, penalty_factors=None, a=A):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = X.shape
    if penalty_factors is None:
        factors = np.ones(p)
    else:
        factors = np.asarray(penalty_factors, dtype=float)
    residual = y - y.mean()
    v = np.sum(X * X, axis=0) / n
    score_num = np.abs(X.T @ residual) / n
    denom = factors * np.minimum(1.0, np.sqrt((a + 1.0) * np.maximum(v, 0.0)))
    raw = np.zeros(p, dtype=float)
    active = (factors > 0.0) & (denom > 0.0)
    raw[active] = score_num[active] / denom[active]
    raw = np.where(np.isfinite(raw), raw, 0.0)
    max_score = float(np.max(raw))
    if max_score == 0.0:
        return 1e-6
    return (1.0 + 1e-4) * max_score


def test_orthogonal_closed_form_matches_distillation():
    n, p = 64, 4
    X = _centered_orthogonal(n, p, seed=0)
    assert np.max(np.abs(X.T @ X / n - np.eye(p))) < 1e-10
    assert np.max(np.abs(X.mean(axis=0))) < 1e-10

    alpha = 1.25
    # At lambda=1 these four slopes sit in the soft-zero, soft-nonzero,
    # middle, and flat pieces. Other grid values move the same slopes
    # through the remaining pieces, including the negative sign.
    beta = np.array([0.8, -1.5, 2.8, -6.0])
    y_tilde = alpha + X @ beta
    lambdas = np.array([2.0, 1.0, 0.3, 0.0])
    regions = set()
    for lam in lambdas:
        for zi in beta:
            az = abs(zi)
            if az <= 2.0 * lam:
                regions.add("soft")
            elif az <= A * lam:
                regions.add("middle")
            else:
                regions.add("flat")
    assert regions == {"soft", "middle", "flat"}

    y_other = y_tilde + np.linspace(-3.0, 3.0, n)
    result = distill(
        X,
        y_other,
        estimator=_fixed_estimator(alpha, beta),
        lambdas=lambdas,
        tol_factor=1e-12,
        max_sweeps=200,
    )
    assert np.allclose(result.y_tilde, y_tilde)
    for i, lam in enumerate(lambdas):
        expected = _closed_form(beta, lam)
        assert np.allclose(theta_scad(beta, lam), expected, atol=1e-12, rtol=0.0)
        assert np.allclose(result.path.beta[i], expected, atol=1e-8, rtol=0.0)
        assert abs(float(result.path.alpha[i]) - alpha) < 1e-8


def test_unpenalized_solve_reproduces_initial():
    rng = np.random.default_rng(2)
    n, p = 48, 5
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    assert np.linalg.matrix_rank(np.column_stack([np.ones(n), X]), tol=1e-10) == p + 1

    fit = ols(X, y)
    result = distill(
        X,
        y,
        estimator="ols",
        lambdas=np.array([0.0]),
        tol_factor=1e-12,
        max_sweeps=20000,
    )
    assert abs(float(result.alpha_hat) - float(fit.alpha)) < 1e-8
    assert np.max(np.abs(np.asarray(result.beta_hat) - np.asarray(fit.beta))) < 1e-8
    assert abs(float(result.path.alpha[0]) - float(fit.alpha)) < 1e-8
    assert np.max(np.abs(np.asarray(result.path.beta[0]) - np.asarray(fit.beta))) < 1e-8


def test_scalar_update_minimizes_conditional():
    a = A
    cases = []
    for v in (0.15, 1.0 / (a - 1.0) - 0.05, 1.0 / (a - 1.0) + 0.05, 0.8, 2.5):
        for zeta in (-5.0, -1.2, -0.2, 0.0, 0.35, 1.7, 4.0):
            for ell in (0.25, 0.9, 1.6):
                for current in (-3.0, -0.4, 0.0, 0.7, 2.2):
                    cases.append((current, zeta, v, ell))

    for current, zeta, v, ell in cases:
        chosen = float(scalar_update(current, zeta, v, ell, a))
        sign = np.sign(zeta)
        specials = [
            0.0,
            sign * ell,
            sign * a * ell,
            current,
            chosen,
            *_feasible_stationary(zeta, v, ell, a),
        ]
        span = max(abs(chosen), abs(current), abs(zeta) / v, a * ell, 1.0) + 3.0
        grid = np.linspace(-span, span, 4001)
        points = np.unique(np.concatenate([grid, np.asarray(specials, dtype=float)]))
        q_min = min(_q(point, zeta, v, ell, a) for point in points)
        q_chosen = _q(chosen, zeta, v, ell, a)
        assert q_chosen <= q_min + 1e-8
        assert q_chosen <= _q(current, zeta, v, ell, a) + 1e-8

    # Unpenalized intercept: ell = 0, update is zeta / v.
    assert abs(float(scalar_update(0.4, 1.8, 2.0, 0.0)) - 0.9) < 1e-12


def test_lambda_max_and_intercept_only_minimum():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 3))
    X[:, 0] *= 0.01
    y = X[:, 1] + 0.2 * rng.normal(size=40)
    factors = np.array([2.0, 0.5, 1.0])

    got = float(null_lambda_max(X, y, penalty_factors=factors))
    expected = _lambda_max_formula(X, y, penalty_factors=factors)
    without_sqrt = _lambda_max_formula_without_sqrt(X, y, factors)
    assert abs(got - expected) < 1e-12
    assert abs(expected - without_sqrt) > 1e-4

    # Column 0 carries the signal, so dropping its penalty factor changes lambda_max.
    X_drop = rng.normal(size=(40, 2))
    y_aligned = 3.0 * X_drop[:, 0] + 0.01 * rng.normal(size=40)
    excluded = np.array([0.0, 1.0])
    got_excluded = float(null_lambda_max(X_drop, y_aligned, penalty_factors=excluded))
    assert abs(got_excluded - _lambda_max_formula(X_drop, y_aligned, penalty_factors=excluded)) < 1e-12
    assert abs(got_excluded - _lambda_max_formula(X_drop, y_aligned, penalty_factors=np.ones(2))) > 1e-3

    y_flat = np.full(40, 3.0)
    assert float(null_lambda_max(X, y_flat)) == pytest.approx(1e-6)

    lam = float(null_lambda_max(X, y))
    n, p = X.shape
    _Z, G, c = gram_system(X, y)
    b = intercept_only_state(y, p)
    f = penalty_factor_vector(p)
    for j in range(p + 1):
        gb_j = float(G[j] @ b)
        vj = float(G[j, j])
        zeta = float(c[j]) - gb_j + vj * float(b[j])
        updated = float(scalar_update(float(b[j]), zeta, vj, lam * float(f[j])))
        assert abs(updated - float(b[j])) < 1e-10
        assert _q(updated, zeta, vj, lam * float(f[j])) <= _q(float(b[j]), zeta, vj, lam * float(f[j])) + 1e-12

    path = scad_path(X, y, n_lambda=3, max_sweeps=50)
    assert abs(float(path.lambdas[0]) - lam) < 1e-12
    assert np.max(np.abs(np.asarray(path.beta[0]))) < 1e-10
    assert abs(float(path.alpha[0]) - float(np.mean(y))) < 1e-12
    assert bool(path.accepted[0])


def _lambda_max_formula_without_sqrt(X, y, factors, a=A):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = X.shape[0]
    residual = y - y.mean()
    v = np.sum(X * X, axis=0) / n
    score_num = np.abs(X.T @ residual) / n
    denom = factors * np.minimum(1.0, (a + 1.0) * np.maximum(v, 0.0))
    raw = np.where((factors > 0.0) & (denom > 0.0), score_num / denom, 0.0)
    return (1.0 + 1e-4) * float(np.max(raw))


def test_gic_selects_fidelity_criterion():
    assert int(degrees_of_freedom(np.zeros(4))) == 1
    assert int(degrees_of_freedom(np.array([1e-10, -1e-10]))) == 1
    assert int(degrees_of_freedom(np.array([1e-10 + 1e-12, -5e-10, 0.0]))) == 3

    rss = np.array([4.0, 1.0])
    sigma = 1e-8
    n = 10
    df = np.array([2.0, 4.0])
    floored = fidelity_gic(rss, sigma, n, df)
    expected = rss / 1e-12 + 2.0 * np.log(n) * df
    assert np.allclose(floored, expected)
    assert np.allclose(fidelity_gic(rss, 2.0, n, df), rss / 4.0 + 2.0 * np.log(n) * df)
    assert np.allclose(fidelity_gic(rss, 2.0, n, df), fidelity_gic(rss, 2.0, n, df, kappa=2.0))

    rng = np.random.default_rng(5)
    n, p = 36, 4
    X = rng.normal(size=(n, p))
    beta = np.array([2.0, -1.5, 0.0, 0.0])
    y = 0.3 + X @ beta + 0.4 * rng.normal(size=n)
    result = distill(X, y, estimator="ols", n_lambda=12, max_sweeps=2000)
    assert result.kappa == 2.0
    y_tilde = np.asarray(result.y_tilde)
    sigma = float(result.sigma_hat)
    scale = max(sigma ** 2, 1e-12)
    rss_d = np.array(
        [
            np.sum((y_tilde - float(alpha) - X @ np.asarray(beta_hat)) ** 2)
            for alpha, beta_hat in zip(result.path.alpha, result.path.beta)
        ]
    )
    df = np.array([1 + np.sum(np.abs(np.asarray(beta_hat)) > 1e-10) for beta_hat in result.path.beta])
    gic = rss_d / scale + 2.0 * np.log(n) * df
    assert np.allclose(result.rss_d, rss_d, rtol=1e-10, atol=1e-8)
    assert np.array_equal(np.asarray(result.df), df)
    assert np.allclose(result.gic, gic, rtol=1e-10, atol=1e-8)
    assert result.index_hat == int(np.argmin(gic))
    assert float(result.lambda_hat) == float(result.path.lambdas[result.index_hat])
    gic_without_intercept = rss_d / scale + 2.0 * np.log(n) * (df - 1)
    assert not np.allclose(result.gic, gic_without_intercept)


def test_stage_two_depends_only_on_fitted_surface():
    rng = np.random.default_rng(6)
    n, p = 40, 4
    X = rng.normal(size=(n, p))
    alpha = -0.4
    beta = np.array([1.0, -0.5, 0.25, 0.0])
    beta_other = np.array([0.0, 1.5, -1.0, 0.8])
    y1 = rng.normal(size=n)
    y2 = rng.normal(size=n) + 4.0
    kwargs = dict(n_lambda=8, max_sweeps=1000, tol_factor=1e-8)
    first = distill(X, y1, estimator=_fixed_estimator(alpha, beta), **kwargs)
    second = distill(X, y2, estimator=_fixed_estimator(alpha, beta), **kwargs)
    other = distill(X, y1, estimator=_fixed_estimator(alpha, beta_other), **kwargs)
    assert np.allclose(first.path.coefficients, second.path.coefficients)
    assert np.allclose(first.path.lambdas, second.path.lambdas)
    assert not np.allclose(first.path.beta, other.path.beta)


def test_safeguard_records_acceptance_and_rejects_worse_objectives():
    assert accept_candidate(1.0, 0.4, True, True) == (True, "")
    assert accept_candidate(1.0, 1.0, True, True) == (True, "")
    rejected, reason = accept_candidate(1.0, 1.0 + 1e-15, True, True)
    assert rejected is False and reason == "objective_increase"
    assert accept_candidate(0.2, 0.1, False, True) == (False, "nonfinite")
    assert accept_candidate(0.2, 0.1, True, False) == (False, "fixed_point")

    rng = np.random.default_rng(7)
    n, p = 30, 3
    X = rng.normal(size=(n, p))
    y = X @ np.array([1.0, -0.5, 0.2]) + 0.3 * rng.normal(size=n)
    stopped = scad_path(X, y, n_lambda=4, max_sweeps=0)
    assert stopped.accepted.shape == (4,)
    assert not np.any(np.asarray(stopped.accepted))
    assert set(stopped.failure_reason) == {"fixed_point"}
    assert len(stopped.failure_reason) == 4
    for name in (
        "iterations",
        "fixed_point_gap",
        "objective_start",
        "objective_candidate",
        "objective_retained",
    ):
        assert np.asarray(getattr(stopped, name)).shape[0] == 4
    null = intercept_only_state(y, p)
    assert np.allclose(stopped.coefficients, null)
    assert np.allclose(stopped.objective_retained, stopped.objective_start)

    path = scad_path(X, y, n_lambda=8, max_sweeps=2000)
    start = np.asarray(path.objective_start)
    candidate = np.asarray(path.objective_candidate)
    retained = np.asarray(path.objective_retained)
    accepted = np.asarray(path.accepted)
    factors = penalty_factor_vector(p)
    warm = null
    for i, lam in enumerate(np.asarray(path.lambdas)):
        stored = path.coefficients[i]
        recomputed = float(full_objective(stored, X, y, lam, factors))
        assert abs(recomputed - float(retained[i])) < 1e-9
        assert recomputed <= float(start[i]) + 1e-12
        assert float(retained[i]) <= float(start[i]) + 1e-15
        if candidate[i] > start[i]:
            assert not accepted[i]
            assert path.failure_reason[i] == "objective_increase"
            assert np.allclose(stored, warm)
        if accepted[i]:
            assert candidate[i] <= start[i]
            assert path.failure_reason[i] == ""
            assert np.isfinite(path.fixed_point_gap[i])
        else:
            assert np.allclose(stored, warm)
        warm = stored


def test_contaminated_huber_distillation_recovers_support():
    # Seed 1 at n=120 recovers {0, 1, 4}; Huber slope error is about 0.36 of OLS.
    X, y, beta0 = contaminated_design(
        n=120, p=8, seed=1, contamination=0.1, shift=8.0, noise_sd=1.0, rho=0.5
    )
    assert X.shape == (120, 8)
    assert int(round(0.1 * 120)) == 12

    huber = huber_m(X, y)
    ordinary = ols(X, y)
    huber_error = float(np.linalg.norm(np.asarray(huber.beta) - beta0))
    ols_error = float(np.linalg.norm(np.asarray(ordinary.beta) - beta0))
    assert huber_error < 0.5 * ols_error
    assert ols_error - huber_error > 0.3

    result = distill(X, y, estimator="huber")
    assert len(result.path.lambdas) == 100
    assert result.kappa == 2.0
    assert list(np.asarray(result.support)) == [0, 1, 4]
    active = np.abs(np.asarray(result.beta_hat)[[0, 1, 4]])
    inactive = np.abs(np.asarray(result.beta_hat)[[2, 3, 5, 6, 7]])
    assert np.min(active) > 1.0
    assert np.max(inactive) <= 1e-8

    y_tilde = np.asarray(huber.fitted)
    y_hat = float(result.alpha_hat) + X @ np.asarray(result.beta_hat)
    surface_scale = np.linalg.norm(y_tilde - y_tilde.mean())
    relative = np.linalg.norm(y_tilde - y_hat) / surface_scale
    assert relative < 0.1

    sigma = float(huber.sigma_hat)
    scale = max(sigma ** 2, 1e-12)
    rss_tilde = np.array(
        [
            np.sum((y_tilde - float(alpha) - X @ np.asarray(beta)) ** 2)
            for alpha, beta in zip(result.path.alpha, result.path.beta)
        ]
    )
    df = np.array([1 + np.sum(np.abs(np.asarray(beta)) > 1e-10) for beta in result.path.beta])
    gic = rss_tilde / scale + 2.0 * np.log(len(y)) * df
    assert np.allclose(result.gic, gic, rtol=1e-10, atol=1e-7)
    assert result.index_hat == int(np.argmin(gic))

    rss_y = np.array(
        [
            np.sum((y - float(alpha) - X @ np.asarray(beta)) ** 2)
            for alpha, beta in zip(result.path.alpha, result.path.beta)
        ]
    )
    gic_y = rss_y / scale + 2.0 * np.log(len(y)) * df
    assert int(np.argmin(gic_y)) != result.index_hat
    assert float(np.max(np.abs(gic_y - gic))) > 100.0


def test_sigma_hat_mad_rms_unit_fallback():
    def mad(residual):
        residual = np.asarray(residual, dtype=float)
        return 1.4826 * np.median(np.abs(residual - np.median(residual)))

    def rms(residual):
        residual = np.asarray(residual, dtype=float)
        return float(np.sqrt(np.mean(residual ** 2)))

    spread = np.array([0.2, -0.5, 1.1, -0.3, 0.7, -1.4, 0.4, 2.0, -0.8])
    assert mad(spread) > 1e-10
    assert rms(spread) > 1e-10
    scale, branch = residual_scale(spread)
    assert branch == "mad"
    assert abs(float(scale) - mad(spread)) < 1e-12

    # More than half the residuals are zero, so MAD is zero while RMS is not.
    spiked = np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    assert mad(spiked) <= 1e-10
    assert rms(spiked) > 1e-10
    scale, branch = residual_scale(spiked)
    assert branch == "rms"
    assert abs(float(scale) - rms(spiked)) < 1e-12

    for residual in (
        np.zeros(5),
        np.array([np.nan, 0.2, -0.2]),
        np.array([np.inf, -np.inf, np.nan]),
    ):
        scale, branch = residual_scale(residual)
        assert branch == "unit"
        assert float(scale) == 1.0
