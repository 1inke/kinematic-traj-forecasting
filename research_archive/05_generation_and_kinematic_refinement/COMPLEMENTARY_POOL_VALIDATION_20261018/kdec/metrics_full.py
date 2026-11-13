"""完整复算：分组质量、分组运动学、聚簇、窗口类别、delta。"""
from __future__ import annotations
import numpy as np

GROUPS = ("overall", "stable", "transition", "change")
QUALITY = (0.2, 0.4)
MOTION_FACTOR = 1.01
MOTION_KEYS = ("speed_p95", "accel_mean", "accel_p95", "jerk_mean", "jerk_p95")
DT = 0.4


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
    good = (ade <= QUALITY[0]) & (fde <= QUALITY[1])
    win = joint.argmin(1)
    rows = np.arange(len(paths))
    return dict(
        ade=ade, fde=fde, joint=joint, good=good,
        winner=win,
        j_closest=joint[rows, win],
        ade_closest=ade[rows, win],
        fde_closest=fde[rows, win],
        path_err=e[rows, win],  # (B,T)
    )


def motion_stats(paths: np.ndarray) -> dict:
    p = paths.astype(np.float64)
    # prepend origin like audit script for consistency with review
    start = np.zeros((*p.shape[:2], 1, 2), dtype=np.float64)
    pos = np.concatenate((start, p), axis=2)
    vel = np.diff(pos, axis=2) / DT
    acc = np.diff(vel, axis=2) / DT
    jerk = np.diff(acc, axis=2) / DT
    s = np.linalg.norm(vel, axis=-1)
    a = np.linalg.norm(acc, axis=-1)
    j = np.linalg.norm(jerk, axis=-1)
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
        r = float(pred[k]) / n if n > 0 else float("inf")
        ratios[k] = r
        if r > factor + 1e-15:
            fails.append(k)
    return dict(status=("PASS" if not fails else "FAIL"), fails=fails, ratios=ratios)


def try_clusters(paths, good):
    """SciPy complete-link; fallback if scipy missing."""
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
    except Exception:
        return None, None
    B, K = paths.shape[:2]
    distinct = np.zeros(B, dtype=np.int32)
    effective = np.zeros(B, dtype=np.int32)
    i, k = np.triu_indices(K, 1)
    for b in range(B):
        row = paths[b]
        d = np.linalg.norm(row[i] - row[k], axis=-1)
        condensed = np.maximum(d.mean(axis=1) / 0.10, d[:, -1] / 0.20)
        if not np.isfinite(condensed).all() or condensed.size == 0:
            distinct[b] = K
            effective[b] = int(good[b].sum())
            continue
        # linkage needs condensed distance vector
        lab = fcluster(linkage(condensed, method="complete"), 1.0, criterion="distance")
        distinct[b] = len(np.unique(lab))
        effective[b] = len(np.unique(lab[good[b]])) if good[b].any() else 0
    return distinct, effective


def evaluate_portfolio(paths, gt, native, groups, do_cluster=True):
    pe = errors(paths, gt)
    ne = errors(native, gt)
    n_cov = ne["good"].any(1)
    p_cov = pe["good"].any(1)
    delta_ade = pe["ade"].min(1) - ne["ade"].min(1)
    delta_fde = pe["fde"].min(1) - ne["fde"].min(1)
    delta_j = pe["j_closest"] - ne["j_closest"]

    distinct = effective = None
    if do_cluster:
        distinct, effective = try_clusters(paths, pe["good"])

    out = {"_per_window": dict(
        covered=p_cov, native_covered=n_cov,
        minADE=pe["ade"].min(1), minFDE=pe["fde"].min(1),
        native_minADE=ne["ade"].min(1), native_minFDE=ne["fde"].min(1),
        j_closest=pe["j_closest"], native_j_closest=ne["j_closest"],
        delta_minADE=delta_ade, delta_minFDE=delta_fde, delta_J=delta_j,
        winner=pe["winner"],
        good_any=p_cov,
    )}
    if distinct is not None:
        out["_per_window"]["distinct"] = distinct
        out["_per_window"]["effective_good"] = effective

    # five window classes relative to native
    retained = n_cov & p_cov
    rescued = (~n_cov) & p_cov
    lost = n_cov & (~p_cov)
    both_fail = (~n_cov) & (~p_cov)
    # union-good-but-selected-fail needs union — filled by caller if provided
    out["_classes"] = dict(
        retained_good=np.where(retained)[0].astype(np.int64),
        rescued=np.where(rescued)[0].astype(np.int64),
        lost=np.where(lost)[0].astype(np.int64),
        both_fail=np.where(both_fail)[0].astype(np.int64),
    )

    for name, sel in group_masks(groups):
        if not sel.any():
            continue
        nm = motion_stats(native[sel])
        pm = motion_stats(paths[sel])
        gate = motion_gate(pm, nm)
        row = dict(
            N=int(sel.sum()),
            coverage=float(p_cov[sel].mean()),
            native_coverage=float(n_cov[sel].mean()),
            good_count=int(p_cov[sel].sum()),
            native_good_count=int(n_cov[sel].sum()),
            rescue=int(rescued[sel].sum()),
            lost=int(lost[sel].sum()),
            minADE=float(pe["ade"][sel].min(1).mean()),
            minFDE=float(pe["fde"][sel].min(1).mean()),
            native_minADE=float(ne["ade"][sel].min(1).mean()),
            native_minFDE=float(ne["fde"][sel].min(1).mean()),
            mean_minJ=float(pe["j_closest"][sel].mean()),
            native_mean_minJ=float(ne["j_closest"][sel].mean()),
            delta_minADE_mean=float(delta_ade[sel].mean()),
            delta_minFDE_mean=float(delta_fde[sel].mean()),
            delta_J_mean=float(delta_j[sel].mean()),
            # tail 10%: contribution of worst 10% deltas (fixed descriptive)
            delta_minADE_worst10_mean=float(np.sort(delta_ade[sel])[-max(1, int(np.ceil(0.1 * sel.sum()))):].mean()),
            delta_minFDE_worst10_mean=float(np.sort(delta_fde[sel])[-max(1, int(np.ceil(0.1 * sel.sum()))):].mean()),
            motion=pm,
            native_motion=nm,
            motion_gate=gate,
        )
        if distinct is not None:
            row["distinct_mean"] = float(distinct[sel].mean())
            row["effective_good_mean"] = float(effective[sel].mean())
        out[name] = row
    return out


def summarize_deltas(delta, mask=None):
    x = delta if mask is None else delta[mask]
    pos = x[x > 0]
    neg = x[x < 0]
    return dict(
        mean=float(x.mean()),
        sum=float(x.sum()),
        pos_sum=float(pos.sum()) if len(pos) else 0.0,
        neg_sum=float(neg.sum()) if len(neg) else 0.0,
        n_pos=int((x > 0).sum()),
        n_neg=int((x < 0).sum()),
        p90=float(np.quantile(x, 0.9)),
        p10=float(np.quantile(x, 0.1)),
    )
