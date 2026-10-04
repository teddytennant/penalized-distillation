"""Penalized distillation: sparse regression from one robust fit.

Implements Algorithm 1 of Shin and Park, arXiv:2609.23937. Stage 2 sees the
response only through the initial fitted surface.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from penalized_distillation.distill import distill, scad_raw
from penalized_distillation.initial import huber_m, ols, residual_scale
from penalized_distillation.path import (
    accept_candidate,
    lambda_grid,
    null_lambda_max,
    scad_path,
)
from penalized_distillation.scad import (
    SCAD_A,
    full_objective,
    scalar_objective,
    scalar_update,
    scad_penalty,
    theta_scad,
)

__all__ = [
    "SCAD_A",
    "accept_candidate",
    "distill",
    "full_objective",
    "huber_m",
    "lambda_grid",
    "null_lambda_max",
    "ols",
    "residual_scale",
    "scalar_objective",
    "scalar_update",
    "scad_path",
    "scad_penalty",
    "scad_raw",
    "theta_scad",
]
