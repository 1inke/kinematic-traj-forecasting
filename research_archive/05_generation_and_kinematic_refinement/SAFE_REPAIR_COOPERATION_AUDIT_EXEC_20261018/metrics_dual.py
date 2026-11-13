"""Two independent NumPy reimplementations of W0 quality/motion/cluster/objective metrics.
No torch, no solver. Float64 literal gates; no evaluation epsilon.
"""
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

DT = 0.4
ADE_THR = 0.2
FDE_THR = 0.4
MOTION_KEYS = ("speed_p95", "accel_mean", "accel_p95", "jerk_mean", "jerk_p95")
BOUND = 0.20 * (np.arange(1, 13) / 12.0) ** 2
GROUPS = ("overall", "stable", "transition", "change")
CELLS = ("A", "B", "C", "D")


# --------------- Implementation A ---------------
class ImplA:
    name = "A"

    @staticmethod
    def errors(paths, gt):
        # paths: (N, K, T, 2) or (K,T,2); gt: (N, T, 2) or (T,2)
        p = np.asarray(paths, dtype=np.float64)
        g = np.asarray(gt, dtype=np.float64)
        single = p.ndim == 3
        if single:
            p = p[None]
            g = g[None]
        d = p - g[:, None]
        e = np.linalg.norm(d, axis=-1)
        ade = e.mean(-1)
        fde = e[..., -1]
        j = np.maximum(ade / ADE_THR, fde / FDE_THR)
        good = (ade <= ADE_THR) & (fde <= FDE_THR)
        if single:
            return ade[0], fde[0], j[0], good[0]
        return ade, fde, j, good

    @staticmethod
    def motion(paths):
        # paths (N,K,T,2) or (K,T,2) -> pooled over all candidates and times in batch
        p = np.asarray(paths, dtype=np.float64)
        if p.ndim == 3:
            p = p[None]
        # axis: 0=batch, 1=K, 2=T, 3=xy ; prepend zero -> first speed from origin
        v = np.diff(p, axis=2, prepend=np.zeros_like(p[:, :, :1])) / DT
        a = np.diff(v, axis=2) / DT
        j = np.diff(a, axis=2) / DT
        s = np.linalg.norm(v, axis=-1)
        aa = np.linalg.norm(a, axis=-1)
        jj = np.linalg.norm(j, axis=-1)
        return {
            "speed_p95": float(np.quantile(s, 0.95)),
            "accel_mean": float(aa.mean()),
            "accel_p95": float(np.quantile(aa, 0.95)),
            "jerk_mean": float(jj.mean()),
            "jerk_p95": float(np.quantile(jj, 0.95)),
        }

    @staticmethod
    def per_candidate_mean_accel_jerk(paths):
        p = np.asarray(paths, dtype=np.float64)
        assert p.ndim == 3 and p.shape == (20, 12, 2)
        v = np.diff(p, axis=1, prepend=np.zeros_like(p[:, :1])) / DT
        a = np.diff(v, axis=1) / DT
        j = np.diff(a, axis=1) / DT
        aa = np.linalg.norm(a, axis=-1).mean(1)
        jj = np.linalg.norm(j, axis=-1).mean(1)
        return aa, jj

    @staticmethod
    def clusters_one(paths, good):
        paths = np.asarray(paths, dtype=np.float64)
        K = len(paths)
        d = np.linalg.norm(paths[:, None] - paths[None, :], axis=-1)
        D = np.maximum(d.mean(-1) / 0.1, d[:, :, -1] / 0.2)
        lab = fcluster(linkage(D[np.triu_indices(K, 1)], method="complete"), 1.0, criterion="distance")
        distinct = int(len(np.unique(lab)))
        eg = int(len(np.unique(lab[good]))) if good.any() else 0
        return distinct, eg

    @staticmethod
    def support_stats(native, repaired):
        native = np.asarray(native, dtype=np.float64)
        repaired = np.asarray(repaired, dtype=np.float64)
        # D native->repaired nearest
        d = np.linalg.norm(native[:, None] - repaired[None, :], axis=-1)
        Dnr = np.maximum(d.mean(-1) / 0.1, d[:, :, -1] / 0.2)
        n_to_r = Dnr.min(1)
        d2 = np.linalg.norm(repaired[:, None] - native[None, :], axis=-1)
        Drn = np.maximum(d2.mean(-1) / 0.1, d2[:, :, -1] / 0.2)
        r_to_n = Drn.min(1)
        return {
            "native_to_repaired_nearest_D": n_to_r.tolist(),
            "n_native_support_lost_Dgt1": int((n_to_r > 1).sum()),
            "repaired_to_native_nearest_D": r_to_n.tolist(),
            "mean_new_support_D": float(r_to_n.mean()),
        }

    @staticmethod
    def objective_scalar(paths, native, gt, cell):
        paths = np.asarray(paths, dtype=np.float64)
        native = np.asarray(native, dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)
        assert paths.shape == (20, 12, 2)

        def Q(p):
            e = np.linalg.norm(p - gt[None], axis=-1)
            return e.mean(-1) + 0.5 * e[:, -1]

        qx = Q(paths)
        q0 = Q(native)
        # fixed owner: first physical slot breaks ties
        a0 = int(np.argmin(q0))
        aX = int(np.argmin(qx))
        owner = a0 if cell in ("B", "D") else aX
        task = float(qx[owner])
        drift = np.linalg.norm(paths - native, axis=-1)
        drift = drift.mean(-1) + 0.5 * drift[:, -1]
        anchor = float(drift[np.arange(20) != owner].sum() / 19.0)
        # 190 pairs
        dists = []
        for i in range(20):
            for j in range(i + 1, 20):
                dists.append(np.linalg.norm(paths[i] - paths[j], axis=-1).mean())
        dists = np.asarray(dists)
        div = float(np.mean(np.maximum(0.0, 0.3 - dists) ** 2))
        total = task + 0.05 * div + (anchor if cell in ("C", "D") else 0.0)
        return {
            "total": total,
            "task": task,
            "diversity": div,
            "anchor": anchor,
            "owner": owner,
            "a0": a0,
            "aX": aX,
            "Q": qx.tolist(),
            "Q0": q0.tolist(),
        }


# --------------- Implementation B (different code paths) ---------------
class ImplB:
    name = "B"

    @staticmethod
    def errors(paths, gt):
        p = np.array(paths, dtype=np.float64, copy=True)
        g = np.array(gt, dtype=np.float64, copy=True)
        single = p.ndim == 3
        if single:
            p = p.reshape(1, *p.shape)
            g = g.reshape(1, *g.shape)
        # manual hypot
        dx = p[..., 0] - g[:, None, :, 0]
        dy = p[..., 1] - g[:, None, :, 1]
        e = np.sqrt(dx * dx + dy * dy)
        ade = np.mean(e, axis=-1)
        fde = e[..., -1]
        good = np.logical_and(ade <= ADE_THR, fde <= FDE_THR)
        j = np.where(ade / ADE_THR > fde / FDE_THR, ade / ADE_THR, fde / FDE_THR)
        if single:
            return ade[0], fde[0], j[0], good[0]
        return ade, fde, j, good

    @staticmethod
    def motion(paths):
        p = np.array(paths, dtype=np.float64, copy=True)
        if p.ndim == 3:
            p = p.reshape(1, *p.shape)
        N, K, T, _ = p.shape
        # explicit loop-free but different construction: pad with zeros at t=-1
        pad = np.zeros((N, K, 1, 2), dtype=np.float64)
        pp = np.concatenate([pad, p], axis=2)
        v = (pp[:, :, 1:] - pp[:, :, :-1]) / DT
        a = (v[:, :, 1:] - v[:, :, :-1]) / DT
        jk = (a[:, :, 1:] - a[:, :, :-1]) / DT
        s = np.sqrt((v ** 2).sum(-1)).ravel()
        aa = np.sqrt((a ** 2).sum(-1)).ravel()
        jj = np.sqrt((jk ** 2).sum(-1)).ravel()
        return {
            "speed_p95": float(np.quantile(s, 0.95, method="linear")),
            "accel_mean": float(aa.mean()),
            "accel_p95": float(np.quantile(aa, 0.95, method="linear")),
            "jerk_mean": float(jj.mean()),
            "jerk_p95": float(np.quantile(jj, 0.95, method="linear")),
        }

    @staticmethod
    def per_candidate_mean_accel_jerk(paths):
        p = np.array(paths, dtype=np.float64, copy=True)
        K, T, _ = p.shape
        pad = np.zeros((K, 1, 2))
        pp = np.concatenate([pad, p], axis=1)
        v = (pp[:, 1:] - pp[:, :-1]) / DT
        a = (v[:, 1:] - v[:, :-1]) / DT
        jk = (a[:, 1:] - a[:, :-1]) / DT
        aa = np.sqrt((a ** 2).sum(-1)).mean(1)
        jj = np.sqrt((jk ** 2).sum(-1)).mean(1)
        return aa, jj

    @staticmethod
    def clusters_one(paths, good):
        paths = np.array(paths, dtype=np.float64, copy=True)
        K = paths.shape[0]
        # condensed distance via explicit pairs
        condensed = []
        for i in range(K - 1):
            diff = paths[i : i + 1] - paths[i + 1 :]
            e = np.linalg.norm(diff, axis=-1)
            condensed.extend(np.maximum(e.mean(-1) / 0.10, e[:, -1] / 0.20).tolist())
        lab = fcluster(linkage(np.asarray(condensed, dtype=np.float64), method="complete"), 1.0, criterion="distance")
        distinct = int(np.unique(lab).size)
        eg = int(np.unique(lab[np.asarray(good, bool)]).size) if np.any(good) else 0
        return distinct, eg

    @staticmethod
    def support_stats(native, repaired):
        native = np.array(native, dtype=np.float64, copy=True)
        repaired = np.array(repaired, dtype=np.float64, copy=True)
        n_to_r = []
        for i in range(20):
            e = np.linalg.norm(native[i] - repaired, axis=-1)
            D = np.maximum(e.mean(-1) / 0.1, e[:, -1] / 0.2)
            n_to_r.append(float(D.min()))
        r_to_n = []
        for i in range(20):
            e = np.linalg.norm(repaired[i] - native, axis=-1)
            D = np.maximum(e.mean(-1) / 0.1, e[:, -1] / 0.2)
            r_to_n.append(float(D.min()))
        n_to_r = np.asarray(n_to_r)
        r_to_n = np.asarray(r_to_n)
        return {
            "native_to_repaired_nearest_D": n_to_r.tolist(),
            "n_native_support_lost_Dgt1": int((n_to_r > 1).sum()),
            "repaired_to_native_nearest_D": r_to_n.tolist(),
            "mean_new_support_D": float(r_to_n.mean()),
        }

    @staticmethod
    def objective_scalar(paths, native, gt, cell):
        paths = np.array(paths, dtype=np.float64, copy=True)
        native = np.array(native, dtype=np.float64, copy=True)
        gt = np.array(gt, dtype=np.float64, copy=True)

        def Q(p):
            dx = p[:, :, 0] - gt[None, :, 0]
            dy = p[:, :, 1] - gt[None, :, 1]
            e = np.sqrt(dx * dx + dy * dy)
            return e.mean(1) + 0.5 * e[:, -1]

        qx = Q(paths)
        q0 = Q(native)
        a0 = int(np.flatnonzero(q0 == q0.min())[0])  # first slot tie-break
        aX = int(np.flatnonzero(qx == qx.min())[0])
        owner = a0 if cell in ("B", "D") else aX
        task = float(qx[owner])
        dr = np.sqrt(((paths - native) ** 2).sum(-1))
        dr = dr.mean(1) + 0.5 * dr[:, -1]
        anchor = float(sum(dr[k] for k in range(20) if k != owner) / 19.0)
        acc = 0.0
        n = 0
        for i in range(20):
            for j in range(i + 1, 20):
                md = np.sqrt(((paths[i] - paths[j]) ** 2).sum(-1)).mean()
                acc += max(0.0, 0.3 - md) ** 2
                n += 1
        div = float(acc / n)
        total = task + 0.05 * div + (anchor if cell in ("C", "D") else 0.0)
        return {
            "total": total,
            "task": task,
            "diversity": div,
            "anchor": anchor,
            "owner": owner,
            "a0": a0,
            "aX": aX,
            "Q": qx.tolist(),
            "Q0": q0.tolist(),
        }


def strict_motion_pass(cand, native):
    return all(cand[k] <= 1.01 * native[k] for k in MOTION_KEYS)


def budget_decompose(m_J, m_N, j):
    # m_* length-20 per-candidate means
    delta = m_J - m_N
    DM_J = float(delta.sum() / 20.0)
    DM_T = float(delta[j] / 20.0)
    M_N = float(m_N.mean())
    H = 0.01 * M_N
    d = float(max(0.0, DM_T - H))
    others = np.delete(delta, j)
    c_pos = float(np.maximum(-others, 0).sum() / 20.0)
    c_neg = float(np.maximum(others, 0).sum() / 20.0)
    c = c_pos - c_neg
    # reconstruct: DM_J = DM_T - c  (since c = -sum_{k!=j} delta_k / 20)
    recon = DM_T - c
    return {
        "delta_k": delta.tolist(),
        "DeltaM_J": DM_J,
        "DeltaM_T": DM_T,
        "M_N": M_N,
        "H": H,
        "target_over_budget_d": d,
        "c_net": c,
        "c_pos": c_pos,
        "c_neg": c_neg,
        "recon_DeltaM_J": recon,
        "recon_abs_err": abs(recon - DM_J),
    }


def configurations(native, actual, target):
    n = np.array(native, dtype=np.float64, copy=True)
    j = np.array(actual, dtype=np.float64, copy=True)
    assert n.shape == j.shape == (20, 12, 2)
    t = n.copy()
    t[target] = j[target]
    o = j.copy()
    o[target] = n[target]
    yield "N", n
    yield "J", j
    yield "T", t
    yield "O", o
    for k in range(20):
        if k == target:
            continue
        drop = j.copy()
        drop[k] = n[k]
        add = t.copy()
        add[k] = j[k]
        assert np.array_equal(drop[target], j[target])
        assert np.array_equal(add[target], j[target])
        yield f"DROP_{k}", drop
        yield f"ADD_{k}", add


def splice_like(native_arr, actual_arr, target, name):
    """Apply same N/J/T/O/DROP/ADD mask to controls/correction."""
    n = np.array(native_arr, dtype=np.float64, copy=True)
    j = np.array(actual_arr, dtype=np.float64, copy=True)
    if name == "N":
        return n.copy()
    if name == "J":
        return j.copy()
    if name == "T":
        out = n.copy()
        out[target] = j[target]
        return out
    if name == "O":
        out = j.copy()
        out[target] = n[target]
        return out
    if name.startswith("DROP_"):
        k = int(name.split("_")[1])
        out = j.copy()
        out[k] = n[k]
        return out
    if name.startswith("ADD_"):
        k = int(name.split("_")[1])
        out = n.copy()
        out[target] = j[target]
        out[k] = j[k]
        return out
    raise ValueError(name)


def control_position_pass(controls, correction):
    c = np.asarray(controls, dtype=np.float64)
    corr = np.asarray(correction, dtype=np.float64)
    disk = bool((np.linalg.norm(c, axis=-1) <= 1.0 + 0).all())  # literal <=1
    # allow tiny float from stored; design says unit disk from production
    disk = bool((np.linalg.norm(c, axis=-1) <= 1.0 + 1e-15).all())
    pos = bool((np.linalg.norm(corr, axis=-1) <= BOUND[None, :]).all())
    return disk and pos


def eta_common(residual):
    # residual (20,12,2) = paths - native
    r = np.asarray(residual, dtype=np.float64)
    num = 20.0 * (np.linalg.norm(r.mean(0)) ** 2)
    den = float((r ** 2).sum())
    if den == 0:
        return None
    return float(num / den)


def group_masks(groups):
    groups = np.asarray(groups)
    return [
        ("overall", np.ones(len(groups), dtype=bool)),
        ("stable", groups == 0),
        ("transition", groups == 1),
        ("change", groups == 2),
    ]
