"""ADE/FDE/good/coverage 与相对 native 的运动学 PASS。"""
from __future__ import annotations
import numpy as np

GROUPS = ("overall", "stable", "transition", "change")
QUALITY = (0.2, 0.4)
MOTION_FACTOR = 1.01
MOTION_KEYS = ("speed_p95", "accel_mean", "accel_p95", "jerk_mean", "jerk_p95")


def masks(groups: np.ndarray):
    return [
        (name, np.ones(len(groups), bool) if j == 0 else groups == (j - 1))
        for j, name in enumerate(GROUPS)
    ]


def errors(p: np.ndarray, gt: np.ndarray):
    d = p.astype(np.float64) - gt[:, None].astype(np.float64)
    e = np.linalg.norm(d, axis=-1)
    a = e.mean(-1)
    f = e[:, :, -1]
    j = np.maximum(a / QUALITY[0], f / QUALITY[1])
    g = (a <= QUALITY[0]) & (f <= QUALITY[1])
    ix = j.argmin(1)
    one = e[np.arange(len(p)), ix]
    return a, f, j, g, one, ix


def motion_stats(p: np.ndarray) -> dict:
    p = p.astype(np.float64)
    v = np.diff(p, axis=2, prepend=np.zeros_like(p[:, :, :1])) / 0.4
    a = np.diff(v, axis=2) / 0.4
    j = np.diff(a, axis=2) / 0.4
    s = np.linalg.norm(v, axis=-1)
    aa = np.linalg.norm(a, axis=-1)
    jj = np.linalg.norm(j, axis=-1)
    return dict(
        speed_p95=float(np.quantile(s, 0.95)),
        accel_mean=float(aa.mean()),
        accel_p95=float(np.quantile(aa, 0.95)),
        jerk_mean=float(jj.mean()),
        jerk_p95=float(np.quantile(jj, 0.95)),
    )


def motion_gate(pred_stats: dict, native_stats: dict, factor: float = MOTION_FACTOR):
    fails = []
    ratios = {}
    for k in MOTION_KEYS:
        n = float(native_stats[k])
        r = float(pred_stats[k]) / n if n > 0 else float("inf")
        ratios[k] = r
        if r > factor:
            fails.append(k)
    return dict(status=("PASS" if not fails else "FAIL"), fails=fails, ratios=ratios)


def evaluate_split(paths: np.ndarray, gt: np.ndarray, native: np.ndarray, groups: np.ndarray):
    ade, fde, joint, good, one, winner = errors(paths, gt)
    n_ade, n_fde, n_joint, n_good, _, _ = errors(native, gt)
    native_m = motion_stats(native)
    pred_m = motion_stats(paths)
    gate = motion_gate(pred_m, native_m)
    out = {}
    for name, sel in masks(groups):
        if not sel.any():
            continue
        g = good[sel]
        ng = n_good[sel]
        covered = g.any(1)
        n_covered = ng.any(1)
        rescue = int((~n_covered & covered).sum())
        lost = int((n_covered & ~covered).sum())
        out[name] = dict(
            N=int(sel.sum()),
            coverage=float(covered.mean()),
            native_coverage=float(n_covered.mean()),
            minADE=float(ade[sel].min(1).mean()),
            minFDE=float(fde[sel].min(1).mean()),
            native_minADE=float(n_ade[sel].min(1).mean()),
            native_minFDE=float(n_fde[sel].min(1).mean()),
            meanADE_winner=float(ade[sel, winner[sel]].mean()),
            meanFDE_winner=float(fde[sel, winner[sel]].mean()),
            rescue=rescue,
            lost=lost,
            good_count=int(covered.sum()),
            native_good_count=int(n_covered.sum()),
        )
    out["_motion"] = dict(pred=pred_m, native=native_m, gate=gate)
    return out
