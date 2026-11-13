"""K0-C: fixed greedy native→KIN swap oracle (policy upper bound, not global opt)."""
from __future__ import annotations
import numpy as np
from .config import N_NAT, N_KIN, K_SELECT, MOTION_FACTOR
from .metrics import errors, motion_stats, motion_gate, gather


def _window_metrics(paths20, gt_i, native20):
    """Single window: paths (20,12,2), gt (12,2), native (20,12,2)."""
    pe = errors(paths20[None], gt_i[None])
    ne = errors(native20[None], gt_i[None])
    return dict(
        covered=bool(pe["good"].any()),
        native_covered=bool(ne["good"].any()),
        minADE=float(pe["minADE"][0]),
        minFDE=float(pe["minFDE"][0]),
        minJ=float(pe["j_closest"][0]),
        native_minADE=float(ne["minADE"][0]),
        native_minFDE=float(ne["minFDE"][0]),
        native_minJ=float(ne["j_closest"][0]),
    )


def greedy_k0c_one(pool40_i, gt_i, native_i):
    """One window. Returns indices (20,), action_trace list, reject stats."""
    # start native 0..19
    selected = list(range(N_NAT))
    used_kin = set()
    removed_native = set()
    trace = []
    rejects = dict(motion=0, monotone=0, no_positive=0)

    def cur_paths(sel):
        return pool40_i[np.array(sel, dtype=np.int64)]

    for step in range(N_KIN):
        cur = _window_metrics(cur_paths(selected), gt_i, native_i)
        best = None  # (key, rem_slot_pos, add_kin, new_sel)
        rem_candidates = [j for j in selected if j < N_NAT]
        add_candidates = [j for j in range(N_NAT, N_NAT + N_KIN) if j not in used_kin]
        if not rem_candidates or not add_candidates:
            break
        for rem in rem_candidates:
            for add in add_candidates:
                new_sel = [add if x == rem else x for x in selected]
                # window motion gate
                mg = motion_gate(
                    motion_stats(cur_paths(new_sel)[None]),
                    motion_stats(native_i[None]),
                )
                if mg["status"] != "PASS":
                    rejects["motion"] += 1
                    continue
                nxt = _window_metrics(cur_paths(new_sel), gt_i, native_i)
                # native success must not be lost
                if cur["native_covered"] and cur["covered"] and (not nxt["covered"]):
                    rejects["monotone"] += 1
                    continue
                if cur["native_covered"] and (not nxt["covered"]):
                    rejects["monotone"] += 1
                    continue
                # deltas as improvements
                d_cov = int(nxt["covered"]) - int(cur["covered"])
                d_j = cur["minJ"] - nxt["minJ"]
                d_ade = cur["minADE"] - nxt["minADE"]
                d_fde = cur["minFDE"] - nxt["minFDE"]
                if d_ade < -1e-15 or d_fde < -1e-15:
                    rejects["monotone"] += 1
                    continue
                if not (d_cov > 0 or d_j > 1e-15):
                    rejects["no_positive"] += 1
                    continue
                key = (d_cov, d_j, d_ade + 0.5 * d_fde, -rem, -add)  # max; slots asc => -slot then take max→ min slot
                # Actually: 降序 deltas, 升序 rem/add.
                # Compare with tuple: higher d_cov better, then d_j, then d_ade+0.5d_fde, then smaller rem, smaller add.
                rank = (d_cov, d_j, d_ade + 0.5 * d_fde, -rem, -add)
                cand = (rank, rem, add, new_sel, dict(d_cov=d_cov, d_j=d_j, d_ade=d_ade, d_fde=d_fde, motion_ratios=mg["ratios"]))
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None:
            break
        _, rem, add, new_sel, info = best
        selected = new_sel
        used_kin.add(add)
        removed_native.add(rem)
        trace.append(dict(step=step, remove_native=int(rem), add_kin=int(add), **info))
    idx = np.array(selected, dtype=np.int64)
    assert len(set(idx.tolist())) == K_SELECT
    return idx, trace, rejects


def run_k0c(pool40, gt, native, groups, do_cluster=False):
    from .metrics import evaluate_set, check_W
    B = pool40.shape[0]
    indices = np.zeros((B, K_SELECT), dtype=np.int64)
    traces = []
    reject_tot = dict(motion=0, monotone=0, no_positive=0)
    for b in range(B):
        idx, tr, rej = greedy_k0c_one(pool40[b], gt[b], native[b])
        indices[b] = idx
        traces.append(tr)
        for k in reject_tot:
            reject_tot[k] += rej[k]
        if (b + 1) % 200 == 0:
            print(f"  K0-C {b+1}/{B}", flush=True)
    paths = gather(pool40, indices)
    ev = evaluate_set(paths, gt, native, groups, do_cluster=do_cluster)
    w = check_W(ev, groups)
    return dict(
        indices=indices,
        paths=paths,
        action_traces=traces,
        reject_totals=reject_tot,
        n_swaps_mean=float(np.mean([len(t) for t in traces])),
        eval=ev,
        W=w,
        tag="ORACLE_DIAGNOSTIC_ONLY",
        note="Fixed greedy policy upper bound; not global action-sequence optimum.",
    )
