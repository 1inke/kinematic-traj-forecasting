"""I / S supervision labels. Positive = improvement. FIT_train GT only."""
from __future__ import annotations
import numpy as np
from .config import QUALITY


def _cand_errors(path, gt):
    """path (12,2), gt (12,2) -> A,F,J,g"""
    e = np.linalg.norm(path.astype(np.float64) - gt.astype(np.float64), axis=-1)
    A = float(e.mean())
    F = float(e[-1])
    J = max(A / QUALITY[0], F / QUALITY[1])
    g = 1.0 if (A <= QUALITY[0] and F <= QUALITY[1] and J <= 1.0) else 0.0
    return A, F, J, g


def set_min_metrics(paths20, gt):
    """paths (20,12,2) -> minA, minF, minJ, c"""
    d = paths20.astype(np.float64) - gt[None].astype(np.float64)
    e = np.linalg.norm(d, axis=-1)
    ade = e.mean(-1)
    fde = e[:, -1]
    joint = np.maximum(ade / QUALITY[0], fde / QUALITY[1])
    good = (ade <= QUALITY[0]) & (fde <= QUALITY[1]) & (joint <= 1.0)
    return float(ade.min()), float(fde.min()), float(joint.min()), float(good.any())


def labels_IS(pool40_i, gt_i, selected, rem, add, native_covered):
    """
    Return dict with I and S 5-vectors (dA,dF,dJ,dC,L).
    native_covered: c(native20) for this window (0/1).
    """
    path_r = pool40_i[rem]
    path_l = pool40_i[add]
    Ar, Fr, Jr, gr = _cand_errors(path_r, gt_i)
    Al, Fl, Jl, gl = _cand_errors(path_l, gt_i)

    sel = list(selected)
    paths_s = pool40_i[np.array(sel, dtype=np.int64)]
    minA, minF, minJ, c = set_min_metrics(paths_s, gt_i)
    new_sel = [add if x == rem else x for x in sel]
    paths_sp = pool40_i[np.array(new_sel, dtype=np.int64)]
    minAp, minFp, minJp, cp = set_min_metrics(paths_sp, gt_i)

    I = np.array([
        Ar - Al,
        Fr - Fl,
        Jr - Jl,
        gl - gr,
        gr * (1.0 - gl),
    ], dtype=np.float64)
    S = np.array([
        minA - minAp,
        minF - minFp,
        minJ - minJp,
        cp - c,
        float(native_covered) * (1.0 - cp),
    ], dtype=np.float64)
    return I, S
