"""K0-B: finite integer feasibility (R necessary, then S sufficient)."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

try:
    import pulp
except ImportError as e:
    pulp = None

from .config import (
    N_NAT, K_SELECT, MOTION_FACTOR, GROUPS,
    W_CHANGE_COV_PP, W_CHANGE_ERR_IMPROVE, W_PROTECT_ERR_FACTOR, W_LOST_FRAC,
)
from .metrics import errors, motion_scalars, motion_stats, gather, evaluate_set, check_W, group_masks


def _p95_params(M: int):
    h = 0.95 * (M - 1)
    k = int(np.floor(h))
    gamma = h - k
    return k, gamma


def _path_counts_le(scalars_bkt: np.ndarray, tau: float) -> np.ndarray:
    """scalars (B,K,T) -> b (B,K) count <= tau."""
    return (scalars_bkt <= tau).sum(axis=-1).astype(np.int64)


def build_instance(pool40, gt, native, groups, mode: str = "R"):
    """Build PuLP problem. mode in {'R','S'}."""
    if pulp is None:
        raise RuntimeError("BLOCKED_SOLVER_BACKEND: pulp not importable")

    e = errors(pool40, gt)
    ne = errors(native, gt)
    B, K = pool40.shape[0], pool40.shape[1]
    assert K == 40
    ade, fde, joint, good = e["ade"], e["fde"], e["joint"], e["good"]
    n_cov = ne["good"].any(1)

    speed, accel, jerk = motion_scalars(pool40)  # (B,40,T)

    masks = {name: sel for name, sel in group_masks(groups)}
    native_motion = {name: motion_stats(native[sel]) for name, sel in masks.items() if sel.any()}

    prob = pulp.LpProblem(f"K0B_{mode}", pulp.LpMaximize)

    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(K)] for i in range(B)]
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(B)]

    # exactly 20
    for i in range(B):
        prob += pulp.lpSum(x[i][j] for j in range(K)) == K_SELECT

    # coverage linking
    for i in range(B):
        gslots = np.where(good[i])[0]
        if len(gslots) == 0:
            prob += c[i] == 0
        else:
            prob += c[i] <= pulp.lpSum(x[i][int(j)] for j in gslots)
            for j in gslots:
                prob += c[i] >= x[i][int(j)]

    # objective: maximize change coverage count
    ch_ix = np.where(masks["change"])[0]
    prob += pulp.lpSum(c[int(i)] for i in ch_ix)

    # W coverage / lost
    for gname in GROUPS:
        sel = masks[gname]
        if not sel.any():
            continue
        ix = np.where(sel)[0]
        nat_good = int(n_cov[sel].sum())
        # lost: sum (n_cov & ~c)
        lost_expr = pulp.lpSum((1 - c[int(i)]) for i in ix if n_cov[i])
        prob += lost_expr <= W_LOST_FRAC * nat_good + 1e-9, f"lost_{gname}"
        if gname == "change":
            need = nat_good + int(np.ceil(W_CHANGE_COV_PP * int(sel.sum())))
            prob += pulp.lpSum(c[int(i)] for i in ix) >= need, "change_cov"
        elif gname in ("overall", "stable", "transition"):
            prob += pulp.lpSum(c[int(i)] for i in ix) >= nat_good, f"cov_{gname}"

    # error witnesses y for ADE/FDE/J
    def add_mean_min_constraints(metric, name, improve_change=True):
        nonlocal prob
        y = [[pulp.LpVariable(f"y{name}_{i}_{j}", cat="Binary") for j in range(K)] for i in range(B)]
        for i in range(B):
            for j in range(K):
                prob += y[i][j] <= x[i][j]
            prob += pulp.lpSum(y[i][j] for j in range(K)) == 1
        for gname in GROUPS:
            sel = masks[gname]
            if not sel.any():
                continue
            ix = np.where(sel)[0]
            mean_expr = pulp.lpSum(metric[i, j] * y[i][j] for i in ix for j in range(K)) / float(len(ix))
            nat_mean = float(ne["minADE" if name == "A" else ("minFDE" if name == "F" else "j_closest")][sel].mean())
            if name == "J":
                # all groups: mean minJ <= native
                prob += mean_expr <= nat_mean + 1e-12, f"minJ_{gname}"
            elif gname == "change" and improve_change:
                lim = nat_mean * (1.0 - W_CHANGE_ERR_IMPROVE)
                prob += mean_expr <= lim + 1e-12, f"{name}_change"
            elif gname in ("overall", "stable", "transition"):
                lim = nat_mean * W_PROTECT_ERR_FACTOR
                prob += mean_expr <= lim + 1e-12, f"{name}_{gname}"
        return y

    yA = add_mean_min_constraints(ade, "A")
    yF = add_mean_min_constraints(fde, "F")
    yJ = add_mean_min_constraints(joint, "J")

    # mean motion (accel_mean, jerk_mean) — exact linear
    for gname in GROUPS:
        sel = masks[gname]
        if not sel.any():
            continue
        ix = np.where(sel)[0]
        Ng = float(len(ix))
        # accel: sum x * sum_t a / (Ng*20*11)
        acc_sum = pulp.lpSum(float(accel[i, j].sum()) * x[i][j] for i in ix for j in range(K))
        prob += acc_sum / (Ng * K_SELECT * 11.0) <= MOTION_FACTOR * native_motion[gname]["accel_mean"] + 1e-15, f"acc_mean_{gname}"
        jerk_sum = pulp.lpSum(float(jerk[i, j].sum()) * x[i][j] for i in ix for j in range(K))
        prob += jerk_sum / (Ng * K_SELECT * 10.0) <= MOTION_FACTOR * native_motion[gname]["jerk_mean"] + 1e-15, f"jerk_mean_{gname}"

    # p95 R/S
    p95_specs = [
        ("speed", speed, 12, "speed_p95"),
        ("accel", accel, 11, "accel_p95"),
        ("jerk", jerk, 10, "jerk_p95"),
    ]
    for gname in GROUPS:
        sel = masks[gname]
        if not sel.any():
            continue
        ix = np.where(sel)[0]
        Ng = int(len(ix))
        for tag, scal, T, mkey in p95_specs:
            M = Ng * K_SELECT * T
            k, gamma = _p95_params(M)
            tau = MOTION_FACTOR * native_motion[gname][mkey]
            # if native is 0, tau=0; counts of <=0
            b = _path_counts_le(scal[ix], tau)  # (Ng, 40)
            if mode == "R":
                need_cnt = k + 1
            else:  # S
                need_cnt = (k + 2) if gamma > 0 else (k + 1)
            expr = pulp.lpSum(int(b[ri, j]) * x[int(ix[ri])][j] for ri in range(Ng) for j in range(K))
            prob += expr >= need_cnt, f"p95_{mode}_{tag}_{gname}"

    meta = dict(
        B=B, K=K, mode=mode,
        n_change=int(masks["change"].sum()),
        native_change_good=int(n_cov[masks["change"]].sum()),
        change_need=int(n_cov[masks["change"]].sum()) + int(np.ceil(W_CHANGE_COV_PP * int(masks["change"].sum()))),
        n_vars_x=B * K,
        n_vars_c=B,
    )
    return prob, x, c, meta


def extract_indices(x, B, K=40):
    idx = np.zeros((B, K_SELECT), dtype=np.int64)
    for i in range(B):
        sel = [j for j in range(K) if pulp.value(x[i][j]) is not None and pulp.value(x[i][j]) > 0.5]
        if len(sel) != K_SELECT:
            # numerical: take top by value
            vals = [(pulp.value(x[i][j]) or 0.0, j) for j in range(K)]
            vals.sort(reverse=True)
            sel = [j for _, j in vals[:K_SELECT]]
        assert len(set(sel)) == K_SELECT, f"window {i} got {sel}"
        idx[i] = np.array(sorted(sel), dtype=np.int64)  # store ascending; order not semantic
    return idx


def solve_instance(prob, x, c, meta, time_limit_s, node_limit, log_path: Path | None = None):
    if pulp is None:
        return dict(status="BLOCKED_SOLVER_BACKEND")
    solvers = pulp.listSolvers(onlyAvailable=True)
    if not solvers:
        return dict(status="BLOCKED_SOLVER_BACKEND", solvers=[])

    kwargs = dict(msg=True, timeLimit=time_limit_s)
    # prefer COIN_CMD / PULP_CBC_CMD
    solver = None
    for name in ("PULP_CBC_CMD", "COIN_CMD"):
        if name in solvers:
            cls = getattr(pulp, name)
            try:
                solver = cls(msg=1, timeLimit=time_limit_s, options=[f"sec {time_limit_s}"])
            except TypeError:
                solver = cls(msg=1, timeLimit=time_limit_s)
            break
    if solver is None:
        solver = pulp.getSolver(solvers[0], msg=1, timeLimit=time_limit_s)

    t0 = time.time()
    # redirect log
    try:
        status_code = prob.solve(solver)
    except Exception as e:
        return dict(
            status="BLOCKED_SOLVER_BACKEND",
            error=f"{type(e).__name__}: {e}",
            solvers=solvers,
            wall_s=time.time() - t0,
        )
    wall = time.time() - t0
    st = pulp.LpStatus.get(status_code, str(status_code))
    obj = pulp.value(prob.objective)

    result = dict(
        pulp_status=st,
        status_code=int(status_code) if isinstance(status_code, (int, np.integer)) else status_code,
        objective=float(obj) if obj is not None else None,
        wall_s=wall,
        time_limit_s=time_limit_s,
        node_limit=node_limit,
        solvers_available=solvers,
        meta=meta,
    )
    if st == "Optimal" or (st == "Not Solved" and obj is not None):
        # incumbent may exist on timeout
        pass
    if obj is not None and st in ("Optimal", "Not Solved"):
        # try extract
        try:
            B = meta["B"]
            idx = extract_indices(x, B)
            result["indices"] = idx
            result["has_incumbent"] = True
        except Exception as e:
            result["has_incumbent"] = False
            result["extract_error"] = f"{type(e).__name__}: {e}"
    else:
        result["has_incumbent"] = False

    if st == "Infeasible":
        result["relaxation_certificate_status"] = "NUMERIC_INFEASIBILITY_UNVERIFIED"
        result["status"] = "NUMERIC_INFEASIBILITY_UNVERIFIED"
    elif result.get("has_incumbent"):
        result["status"] = "INCUMBENT"
    elif st == "Optimal":
        result["status"] = "OPTIMAL_NO_EXTRACT"
    else:
        result["status"] = f"SOLVER_{st.upper().replace(' ', '_')}"

    if log_path is not None:
        slim = {k: v for k, v in result.items() if k != "indices"}
        log_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def verify_solution(pool40, gt, native, groups, indices):
    paths = gather(pool40, indices)
    # ensure paths exactly match pool gather
    rows = np.arange(len(indices))[:, None]
    assert np.allclose(paths, pool40[rows, indices])
    ev = evaluate_set(paths, gt, native, groups, do_cluster=False)
    w = check_W(ev, groups)
    return dict(eval=ev, W=w, paths=paths, indices=indices, tag="ORACLE_DIAGNOSTIC_ONLY")


def preflight_64(pool40_fit, gt_fit, native_fit, groups_fit, out_dir: Path, time_limit_s=120):
    """64 FIT_train windows not selected by GT difficulty."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261018)
    B = pool40_fit.shape[0]
    n = min(64, B)
    ix = np.sort(rng.choice(B, size=n, replace=False))
    p = pool40_fit[ix]
    g = gt_fit[ix]
    nat = native_fit[ix]
    grp = groups_fit[ix]
    # ensure all groups present if possible — still not GT-difficulty based
    t0 = time.time()
    try:
        prob, x, c, meta = build_instance(p, g, nat, grp, mode="R")
        build_s = time.time() - t0
        res = solve_instance(prob, x, c, meta, time_limit_s=time_limit_s, node_limit=None,
                             log_path=out_dir / "preflight_R_solve.json")
        res["build_s"] = build_s
        res["n_windows"] = int(n)
        # scale estimate
        scale = 1035 / float(n)
        res["outer_scale_factor"] = scale
        res["est_outer_build_s"] = build_s * scale
        res["est_outer_wall_hint_s"] = (res.get("wall_s") or time_limit_s) * scale
    except Exception as e:
        res = dict(status="BLOCKED_SOLVER_BACKEND", error=f"{type(e).__name__}: {e}")
    (out_dir / "preflight_64.json").write_text(json.dumps({
        k: v for k, v in res.items() if k != "indices"
    }, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)), encoding="utf-8")
    return res
