"""Follow-up: relative set-NN gain; accept only rel>0 + motion PASS."""
from __future__ import annotations
import numpy as np
from gtf_select import pairwise_D_matrix, enumerate_legal, N_NAT, K_SELECT, MAX_SWAPS


def rel_score(r, l, selected, pool40_i, D40):
    others = [x for x in selected if x != r]
    if not others:
        return (0.0, 0.0, -int(r), -int(l)), 0.0
    o = np.asarray(others, dtype=np.int64)
    d_l = float(D40[l, o].min())
    d_r = float(D40[r, o].min())
    rel = d_l - d_r
    return (rel, d_l, -int(r), -int(l)), rel


def select_one_rel(pool40_i, native_i, motion_stats, motion_gate):
    selected = list(range(N_NAT))
    trace = []
    rejects = dict(motion=0, rel_nonpos=0, no_candidate=0)
    native_motion = motion_stats(native_i[None])
    D40 = pairwise_D_matrix(pool40_i)

    for step in range(MAX_SWAPS):
        pairs = enumerate_legal(selected)
        if not pairs:
            rejects["no_candidate"] += 1
            break
        scored = []
        for r, a in pairs:
            key, rel = rel_score(r, a, selected, pool40_i, D40)
            if rel <= 0:
                rejects["rel_nonpos"] += 1
                continue
            scored.append((key, r, a, rel))
        if not scored:
            rejects["no_candidate"] += 1
            break
        scored.sort(key=lambda t: t[0], reverse=True)
        chosen = None
        motion_fail = 0
        for key, r, a, rel in scored:
            new_sel = [a if x == r else x for x in selected]
            paths = pool40_i[np.asarray(new_sel, dtype=np.int64)]
            mg = motion_gate(motion_stats(paths[None]), native_motion)
            if mg["status"] != "PASS":
                motion_fail += 1
                continue
            chosen = (r, a, mg, key, rel)
            break
        rejects["motion"] += motion_fail
        if chosen is None:
            rejects["no_candidate"] += 1
            break
        r, a, mg, key, rel = chosen
        selected = [a if x == r else x for x in selected]
        trace.append(dict(
            step=step, remove_native=int(r), add_kin=int(a),
            rel=float(rel), score_min_D_l=float(key[1]),
            motion_ratios=mg["ratios"],
        ))
    idx = np.asarray(selected, dtype=np.int64)
    assert len(set(idx.tolist())) == K_SELECT
    return idx, trace, rejects


def select_batch_rel(pool40, native, motion_stats, motion_gate, progress_every=100):
    B = pool40.shape[0]
    indices = np.zeros((B, K_SELECT), dtype=np.int64)
    traces = []
    reject_tot = dict(motion=0, rel_nonpos=0, no_candidate=0)
    for b in range(B):
        idx, tr, rej = select_one_rel(pool40[b], native[b], motion_stats, motion_gate)
        indices[b] = idx
        traces.append(tr)
        for k in reject_tot:
            reject_tot[k] += rej[k]
        if progress_every and (b + 1) % progress_every == 0:
            print(f"    GTF_REL {b+1}/{B} swaps_mean={np.mean([len(t) for t in traces]):.2f}", flush=True)
    return indices, traces, reject_tot
