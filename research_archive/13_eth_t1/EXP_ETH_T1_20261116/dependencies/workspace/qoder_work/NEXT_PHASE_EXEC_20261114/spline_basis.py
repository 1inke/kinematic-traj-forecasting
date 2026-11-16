"""Linear interpolation matrices for the E2 curve decoder (numpy/scipy only).

The h3 decoder places three learned anchors ON the path at future steps 4, 8, 12
(relative to the last observed position, which is step 0) and joins them with
straight segments.  E2 keeps exactly the same anchors and only changes how they
are joined:

    linear   : original h3 piecewise-linear interpolation (B0)
    natural  : cubic spline through (0, a4, a8, a12), S''(0)=0, S''(12)=0
    clamped  : cubic spline through the same points, S'(0)=v0, S''(12)=0

Every variant is linear in its inputs, so each becomes a fixed matrix:

    rel[t] = sum_j W[t, j] * anchor_j  +  w_v[t] * v0          (t = 1..12)

where v0 is the last observed per-step displacement (metres per 0.4 s step) in
the same frame as the anchors.  w_v is all zeros for linear/natural.

The matrices are built once (float64), stored as buffers and applied with one
einsum; no learnable parameter is added by the basis itself.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

KNOTS = np.array([0.0, 4.0, 8.0, 12.0])
STEPS = np.arange(1, 13, dtype=np.float64)
VARIANTS = ("linear", "natural", "clamped")


def _linear_matrix(anchor_steps=(4, 8, 12), pred_len=12) -> np.ndarray:
    """Same matrix as hierarchical_decoder_models.interpolation_weights."""
    w = np.zeros((pred_len, len(anchor_steps)))
    prev = 0
    for seg, step in enumerate(anchor_steps):
        for t in range(prev + 1, step + 1):
            alpha = (t - prev) / (step - prev)
            w[t - 1, seg] = alpha
            if seg:
                w[t - 1, seg - 1] = 1.0 - alpha
        prev = step
    return w


def _spline_eval(values: np.ndarray, start_slope: float | None, t=STEPS, nu=0):
    if start_slope is None:
        bc = ("natural", "natural")
    else:
        bc = ((1, float(start_slope)), (2, 0.0))
    return CubicSpline(KNOTS, values, bc_type=bc)(t, nu)


def basis(variant: str):
    """Return (W[12,3], w_v[12]) in float64."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if variant == "linear":
        return _linear_matrix(), np.zeros(12)
    slope0 = None if variant == "natural" else 0.0
    W = np.zeros((12, 3))
    for j in range(3):
        y = np.zeros(4)
        y[j + 1] = 1.0
        W[:, j] = _spline_eval(y, slope0)
    w_v = np.zeros(12)
    if variant == "clamped":
        w_v = _spline_eval(np.zeros(4), 1.0)
    return W, w_v


def apply_basis(anchors: np.ndarray, v0: np.ndarray, variant: str) -> np.ndarray:
    """anchors [..., 3, 2] relative to last obs; v0 [..., 2] per-step -> rel path [..., 12, 2]."""
    W, w_v = basis(variant)
    rel = np.einsum("tj,...jc->...tc", W, anchors)
    return rel + w_v[:, None] * v0[..., None, :]


if __name__ == "__main__":
    for name in VARIANTS:
        W, w_v = basis(name)
        print(name, "W rows (t=1,4,8,12):", np.round(W[[0, 3, 7, 11]], 4).tolist())
        print(name, "w_v:", np.round(w_v, 4).tolist())
