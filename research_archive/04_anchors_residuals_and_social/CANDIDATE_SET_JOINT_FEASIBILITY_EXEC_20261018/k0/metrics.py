"""Independent quality + kinematics. Literal 1.01 gate, no epsilon."""
from __future__ import annotations
import numpy as np
from .config import QUALITY, MOTION_FACTOR, MOTION_KEYS, DT, GROUPS


def group_masks(groups: np.ndarray):
    return [
        (name, np.ones(len(groups), bool) if j == 0 else groups == (j - 1))
        for j, name in enumerate(GROUPS)
    ]


def errors(paths, gt):
    d = paths.astype(np.float64) - gt[:, None].astype(np.float64)
    e = np.linalg.norm(d, axis=-1)
    ade = e.mean(-1)
    fde = e[:, :, -1]
    joint = np.maximum(ade / QUALITY[0], fde / QUALITY[1])
    good = (ade <= QUALITY[0] + 0.0) & (fde <= QUALITY[1] + 0.0)
    # also J<=1 is equivalent
    good = good & (joint <= 1.0)
    win = joint.argmin(1)
    rows = np.arange(len(paths))
    return dict(
        ade=ade, fde=fde, joint=joint, good=good,
        winner=win,
        minADE=ade.min(1), minFDE=fde.min(1),
        j_closest=joint[rows, win],
        ade_closest=ade[rows, win],
        fde_closest=fde[rows, win],
    )


def motion_scalars(paths: np.ndarray):
    """Return per-candidate speed/accel/jerk scalar arrays.
    paths: (B,K,12,2) -> speed (B,K,12), accel (B,K,11), jerk (B,K,10)
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


def motion_stats(paths: np.ndarray) -> dict:
    s, a, j = motion_scalars(paths)
    return dict(
        speed_p95=float(np.quantile(s, 0.95)),
        accel_mean=float(a.mean()),
        accel_p95=float(np.quantile(a, 0.95)),
        jerk_mean=float(j.mean()),
        jerk_p95=float(np.quantile(j, 0.95)),
    )


def motion_gate(pred, native, factor=MOTION_FACTOR):
    fails, ratios = [], {}
    for k in MOTION_KEYS:
        n = float(native[k])
        p = float(pred[k])
        if n == 0.0:
            # literal absolute: pass iff pred==0 (or <=0 which is never for norms)
            ok = p <= 0.0 + 0.0  # exact; norms >=0 so p==0
            ratios[k] = 0.0 if p == 0.0 else float("inf")
            if not ok:
                fails.append(k)
        else:
            r = p / n
            ratios[k] = float(r)
            if r > factor:  # literal, no epsilon
                fails.append(k)
    return dict(status=("PASS" if not fails else "FAIL"), fails=fails, ratios=ratios)


def window_motion_gate(sel_paths, nat_paths):
    """Per-window pooled over its K candidates."""
    return motion_gate(motion_stats(sel_paths[None, ...]), motion_stats(nat_paths[None, ...]))


def gather(pool40, indices):
    """pool40 (B,40,12,2), indices (B,20) -> (B,20,12,2)"""
    B = pool40.shape[0]
    rows = np.arange(B)[:, None]
    return pool40[rows, indices]


def evaluate_set(paths, gt, native, groups, do_cluster=False):
    pe = errors(paths, gt)
    ne = errors(native, gt)
    n_cov = ne["good"].any(1)
    p_cov = pe["good"].any(1)
    out = {"_per_window": dict(
        covered=p_cov, native_covered=n_cov,
        minADE=pe["minADE"], minFDE=pe["minFDE"],
        native_minADE=ne["minADE"], native_minFDE=ne["minFDE"],
        j_closest=pe["j_closest"], native_j_closest=ne["j_closest"],
    )}
    out["_classes"] = dict(
        retained_good=np.where(n_cov & p_cov)[0].astype(np.int64),
        rescued=np.where((~n_cov) & p_cov)[0].astype(np.int64),
        lost=np.where(n_cov & (~p_cov))[0].astype(np.int64),
        both_fail=np.where((~n_cov) & (~p_cov))[0].astype(np.int64),
    )
    if do_cluster:
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            B, K = paths.shape[:2]
            distinct = np.zeros(B, dtype=np.int32)
            effective = np.zeros(B, dtype=np.int32)
            ii, kk = np.triu_indices(K, 1)
            for b in range(B):
                row = paths[b]
                d = np.linalg.norm(row[ii] - row[kk], axis=-1)
                condensed = np.maximum(d.mean(axis=1) / 0.10, d[:, -1] / 0.20)
                if not np.isfinite(condensed).all() or condensed.size == 0:
                    distinct[b] = K
                    effective[b] = int(pe["good"][b].sum())
                    continue
                lab = fcluster(linkage(condensed, method="complete"), 1.0, criterion="distance")
                distinct[b] = len(np.unique(lab))
                effective[b] = len(np.unique(lab[pe["good"][b]])) if pe["good"][b].any() else 0
            out["_per_window"]["distinct"] = distinct
            out["_per_window"]["effective_good"] = effective
        except Exception:
            pass
    for name, sel in group_masks(groups):
        if not sel.any():
            continue
        nm = motion_stats(native[sel])
        pm = motion_stats(paths[sel])
        gate = motion_gate(pm, nm)
        n_good = int(n_cov[sel].sum())
        p_good = int(p_cov[sel].sum())
        lost = int((n_cov[sel] & (~p_cov[sel])).sum())
        row = dict(
            N=int(sel.sum()),
            coverage=float(p_cov[sel].mean()),
            native_coverage=float(n_cov[sel].mean()),
            good_count=p_good,
            native_good_count=n_good,
            lost=lost,
            conditional_lost_frac=(float(lost) / n_good) if n_good else None,
            rescue=int(((~n_cov[sel]) & p_cov[sel]).sum()),
            minADE=float(pe["minADE"][sel].mean()),
            minFDE=float(pe["minFDE"][sel].mean()),
            native_minADE=float(ne["minADE"][sel].mean()),
            native_minFDE=float(ne["minFDE"][sel].mean()),
            mean_minJ=float(pe["j_closest"][sel].mean()),
            native_mean_minJ=float(ne["j_closest"][sel].mean()),
            motion=pm,
            native_motion=nm,
            motion_gate=gate,
        )
        out[name] = row
    return out


def check_W(ev: dict, groups: np.ndarray) -> dict:
    """K0 screening W from 00 §5. Returns pass/fail + per-clause."""
    from .config import (
        W_CHANGE_COV_PP, W_CHANGE_ERR_IMPROVE, W_PROTECT_ERR_FACTOR, W_LOST_FRAC,
    )
    clauses = {}
    ok = True
    ch = ev.get("change")
    if ch is None:
        return dict(pass_=False, clauses={"change_present": False})
    N_ch = ch["N"]
    need = ch["native_good_count"] + int(np.ceil(W_CHANGE_COV_PP * N_ch))
    c1 = ch["good_count"] >= need
    clauses["change_cov"] = dict(need=need, got=ch["good_count"], pass_=c1)
    ok &= c1

    for g in ("overall", "stable", "transition"):
        row = ev[g]
        cov_ok = row["good_count"] >= row["native_good_count"]
        lost_frac = row["conditional_lost_frac"]
        lost_ok = True if lost_frac is None else (lost_frac <= W_LOST_FRAC + 1e-15)
        clauses[f"{g}_cov_ge_native"] = dict(pass_=cov_ok, good=row["good_count"], native=row["native_good_count"])
        clauses[f"{g}_lost"] = dict(pass_=lost_ok, frac=lost_frac, lost=row["lost"], native_good=row["native_good_count"])
        ok &= cov_ok and lost_ok

    # change lost also ≤1%
    lost_frac = ch["conditional_lost_frac"]
    lost_ok = True if lost_frac is None else (lost_frac <= W_LOST_FRAC + 1e-15)
    clauses["change_lost"] = dict(pass_=lost_ok, frac=lost_frac)
    ok &= lost_ok

    # change ADE/FDE improve ≥5%
    ade_ok = ch["minADE"] <= ch["native_minADE"] * (1.0 - W_CHANGE_ERR_IMPROVE) + 1e-15
    fde_ok = ch["minFDE"] <= ch["native_minFDE"] * (1.0 - W_CHANGE_ERR_IMPROVE) + 1e-15
    clauses["change_ade"] = dict(pass_=ade_ok, got=ch["minADE"], need=ch["native_minADE"] * (1 - W_CHANGE_ERR_IMPROVE))
    clauses["change_fde"] = dict(pass_=fde_ok, got=ch["minFDE"], need=ch["native_minFDE"] * (1 - W_CHANGE_ERR_IMPROVE))
    ok &= ade_ok and fde_ok

    for g in ("overall", "stable", "transition"):
        row = ev[g]
        a_ok = row["minADE"] <= row["native_minADE"] * W_PROTECT_ERR_FACTOR + 1e-15
        f_ok = row["minFDE"] <= row["native_minFDE"] * W_PROTECT_ERR_FACTOR + 1e-15
        clauses[f"{g}_ade_protect"] = dict(pass_=a_ok, got=row["minADE"], lim=row["native_minADE"] * W_PROTECT_ERR_FACTOR)
        clauses[f"{g}_fde_protect"] = dict(pass_=f_ok, got=row["minFDE"], lim=row["native_minFDE"] * W_PROTECT_ERR_FACTOR)
        ok &= a_ok and f_ok

    for g in GROUPS:
        row = ev[g]
        j_ok = row["mean_minJ"] <= row["native_mean_minJ"] + 1e-15
        clauses[f"{g}_minJ"] = dict(pass_=j_ok, got=row["mean_minJ"], native=row["native_mean_minJ"])
        ok &= j_ok
        mg = row["motion_gate"]
        m_ok = mg["status"] == "PASS"
        clauses[f"{g}_motion"] = dict(pass_=m_ok, fails=mg["fails"], ratios=mg["ratios"])
        ok &= m_ok

    return dict(pass_=bool(ok), clauses=clauses)
