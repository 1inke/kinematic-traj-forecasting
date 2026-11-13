"""Kinematic metrics and physical gate evaluation for Q0 Action Identifiability Diagnostic.
Strictly follows 01_Q0_实验设计.md Section 5 & k0/metrics.py. Literal 1.01 gate, no epsilon.
Optimized for high-throughput action enumeration.
"""
from __future__ import annotations

import numpy as np

DT = 0.4
MOTION_FACTOR = 1.01
MOTION_KEYS = ["speed_p95", "accel_mean", "accel_p95", "jerk_mean", "jerk_p95"]


def motion_scalars(paths: np.ndarray):
    """Computes speed, accel, and jerk norms.
    paths: (B, K, 12, 2)
    Returns: vel_norm (B, K, 12), acc_norm (B, K, 11), jerk_norm (B, K, 10)
    """
    p = paths.astype(np.float64)
    start = np.zeros((*p.shape[:2], 1, 2), dtype=np.float64)
    pos = np.concatenate((start, p), axis=2)
    vel = np.diff(pos, axis=2) / DT
    acc = np.diff(vel, axis=2) / DT
    jerk = np.diff(acc, axis=2) / DT
    return (
        np.linalg.norm(vel, axis=-1),
        np.linalg.norm(acc, axis=-1),
        np.linalg.norm(jerk, axis=-1),
    )


def motion_stats(paths: np.ndarray) -> dict[str, float]:
    """Computes pooled scalar stats over a candidate set."""
    s, a, j = motion_scalars(paths)
    return {
        "speed_p95": float(np.quantile(s, 0.95)),
        "accel_mean": float(a.mean()),
        "accel_p95": float(np.quantile(a, 0.95)),
        "jerk_mean": float(j.mean()),
        "jerk_p95": float(np.quantile(j, 0.95)),
    }


def precompute_pool_motion(pool40: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precomputes speed, accel, and jerk norms for all 40 candidates in a window.
    Returns: s40 (40, 12), a40 (40, 11), j40 (40, 10)
    """
    s, a, j = motion_scalars(pool40[None, ...])
    return s[0], a[0], j[0]


def subset_motion_stats(s40: np.ndarray, a40: np.ndarray, j40: np.ndarray, selected: list | np.ndarray) -> dict[str, float]:
    """Computes pooled stats for a 20-candidate subset using precomputed norms."""
    sel = np.asarray(selected, dtype=np.int64)
    s_sel = s40[sel]
    a_sel = a40[sel]
    j_sel = j40[sel]
    return {
        "speed_p95": float(np.quantile(s_sel, 0.95)),
        "accel_mean": float(a_sel.mean()),
        "accel_p95": float(np.quantile(a_sel, 0.95)),
        "jerk_mean": float(j_sel.mean()),
        "jerk_p95": float(np.quantile(j_sel, 0.95)),
    }


def motion_gate(pred: dict[str, float], native: dict[str, float], factor: float = MOTION_FACTOR) -> dict:
    fails = []
    ratios = {}
    for k in MOTION_KEYS:
        n = float(native[k])
        p = float(pred[k])
        if n == 0.0:
            ok = (p == 0.0)
            ratios[k] = 0.0 if p == 0.0 else float("inf")
            if not ok:
                fails.append(k)
        else:
            r = p / n
            ratios[k] = float(r)
            if r > factor:
                fails.append(k)
    return {
        "status": "PASS" if not fails else "FAIL",
        "fails": fails,
        "ratios": ratios,
    }


def window_motion_gate(proposed_paths: np.ndarray, native_paths: np.ndarray, factor: float = MOTION_FACTOR) -> dict:
    """Evaluates window-level motion gate of proposed K20 relative to native K20."""
    pred_stats = motion_stats(proposed_paths[None, ...])
    native_stats = motion_stats(native_paths[None, ...])
    return motion_gate(pred_stats, native_stats, factor=factor)
