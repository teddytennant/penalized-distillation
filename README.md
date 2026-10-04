# Penalized distillation

Sparse regression distilled from a single robust fit, following Shin and Park, arXiv:2609.23937.

## Install

```bash
pip install -e .
```

## Demo

```bash
python -m penalized_distillation.demo
```

The demo fits D-Huber and raw-response SCAD on one seeded contaminated design and prints support, fidelity RSS, and slope error. Library grid defaults are the paper's 100-point geometric path (lower endpoint `1e-3 * lambda_max` when `n > p + 1`, otherwise `0.05 * lambda_max`). The demo uses a shorter path.

## Departures from the paper

The second-stage solver, distillation GIC, residual scale, `lambda_max`, and the Huber initial fit follow Section 2 and Appendix D. Huber is IRLS with the MASS::rlm Huber settings used in the paper (psi constant 1.345, center-0 MAD scale divided by 0.6745, 200 iterations, residual convergence 1e-4). It is not a claim of bit-identity with R, and it is not MASS::rlm MM.

Not included: MASS::rlm MM, any simplified MM substitute, the Appendix D.3 failure cascade, the Appendix D.4 weighted MCD variant, the Section 5 high-p screening step, and the superconductivity study. Arrays are float64.
