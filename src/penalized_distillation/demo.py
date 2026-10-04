"""Tiny demo: D-Huber and raw-response SCAD on one contaminated design.

Library defaults are the paper's 100-point grid. This demo uses a short path.
"""

from __future__ import annotations

import numpy as np

from penalized_distillation.design import contaminated_design
from penalized_distillation.distill import distill, scad_raw
from penalized_distillation.initial import ols


def _support_list(indices):
    return [int(i) for i in np.asarray(indices)]


def _slope_error(beta, beta0):
    return float(np.linalg.norm(np.asarray(beta) - np.asarray(beta0)))


def main():
    seed = 7
    n = 160
    n_lambda = 25
    X, y, beta0 = contaminated_design(n=n, seed=seed)
    distilled = distill(X, y, estimator="huber", n_lambda=n_lambda, max_sweeps=2000)
    raw = scad_raw(X, y, n_lambda=n_lambda, max_sweeps=2000)
    fidelity = float(distilled.rss_d[distilled.index_hat])
    print(f"seed={seed} n={n} p={X.shape[1]} n_lambda={n_lambda}")
    print(f"beta0 = {beta0.tolist()}")
    print(f"OLS initial slope error: {_slope_error(ols(X, y).beta, beta0):.6f}")
    print(f"Huber initial slope error: {_slope_error(distilled.beta_tilde, beta0):.6f}")
    print(f"D-Huber support: {_support_list(distilled.support)}")
    print(f"D-Huber fidelity RSS: {fidelity:.6f}")
    print(f"D-Huber slope error: {_slope_error(distilled.beta_hat, beta0):.6f}")
    print(f"SCAD-raw support: {_support_list(raw.support)}")
    print(f"SCAD-raw RSS: {float(raw.rss[raw.index_hat]):.6f}")
    print(f"SCAD-raw slope error: {_slope_error(raw.beta_hat, beta0):.6f}")


if __name__ == "__main__":
    main()
