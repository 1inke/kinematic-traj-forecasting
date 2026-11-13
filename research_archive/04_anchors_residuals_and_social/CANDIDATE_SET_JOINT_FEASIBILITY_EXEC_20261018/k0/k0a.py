"""K0-A: quality upper-bound construction (≤3 quality witnesses + native fill)."""
from __future__ import annotations
import numpy as np
from .config import N_NAT, K_SELECT
from .metrics import errors, gather, evaluate_set, check_W


def select_k0a(pool40: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return (B,20) pool indices. argmin ADE/FDE/J union, ties->min index; fill native-first."""
    e = errors(pool40, gt)
    B = pool40.shape[0]
    out = np.zeros((B, K_SELECT), dtype=np.int64)
    for b in range(B):
        chosen = []
        for key in ("ade", "fde", "joint"):
            row = e[key][b]
            j = int(np.argmin(row))  # min index on ties
            if j not in chosen:
                chosen.append(j)
        # fill native preferred, then remaining by ascending index
        for j in range(N_NAT):
            if len(chosen) >= K_SELECT:
                break
            if j not in chosen:
                chosen.append(j)
        for j in range(pool40.shape[1]):
            if len(chosen) >= K_SELECT:
                break
            if j not in chosen:
                chosen.append(j)
        out[b] = np.array(chosen[:K_SELECT], dtype=np.int64)
    # uniqueness
    assert all(len(set(out[b].tolist())) == K_SELECT for b in range(B))
    return out


def verify_quality_match_union(pool40, gt, indices) -> dict:
    e40 = errors(pool40, gt)
    paths = gather(pool40, indices)
    e20 = errors(paths, gt)
    return dict(
        ade_match=bool(np.allclose(e40["minADE"], e20["minADE"])),
        fde_match=bool(np.allclose(e40["minFDE"], e20["minFDE"])),
        j_match=bool(np.allclose(e40["j_closest"], e20["j_closest"])),
        cov_match=bool(np.array_equal(e40["good"].any(1), e20["good"].any(1))),
        max_ade_diff=float(np.max(np.abs(e40["minADE"] - e20["minADE"]))),
        max_fde_diff=float(np.max(np.abs(e40["minFDE"] - e20["minFDE"]))),
        max_j_diff=float(np.max(np.abs(e40["j_closest"] - e20["j_closest"]))),
    )


def run_k0a(pool40, gt, native, groups, do_cluster=False):
    idx = select_k0a(pool40, gt)
    paths = gather(pool40, idx)
    match = verify_quality_match_union(pool40, gt, idx)
    # union40 eval for quality upper bound
    ev_u = evaluate_set(pool40, gt, native, groups, do_cluster=False)
    w_u = check_W(ev_u, groups)
    ev = evaluate_set(paths, gt, native, groups, do_cluster=do_cluster)
    w = check_W(ev, groups)
    quality_ub_negation = (not w_u["pass_"])  # if union fails quality parts of W
    # more precise: which quality clauses fail on union
    q_fail = []
    for k, v in w_u["clauses"].items():
        if k.endswith("_motion"):
            continue
        if isinstance(v, dict) and not v.get("pass_", True):
            q_fail.append(k)
    return dict(
        indices=idx,
        paths=paths,
        match_union_quality=match,
        eval=ev,
        W=w,
        union_eval=ev_u,
        union_W=w_u,
        quality_upper_bound_negation=bool(q_fail),  # quality-only failures on union
        union_quality_fail_clauses=q_fail,
        tag="ORACLE_DIAGNOSTIC_ONLY",
    )
