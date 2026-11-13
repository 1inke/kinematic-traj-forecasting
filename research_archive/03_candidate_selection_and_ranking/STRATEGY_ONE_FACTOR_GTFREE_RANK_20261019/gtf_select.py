"""GT-free geometric rank selector. Same action space as K0-C; no GT / HGB."""
from __future__ import annotations
import numpy as np

N_NAT, N_KIN, K_SELECT, MAX_SWAPS = 20, 20, 20, 20


def pairwise_D_matrix(pool40_i: np.ndarray) -> np.ndarray:
    """D = max(meanpath/0.10, endpoint/0.20). pool40_i: (40,12,2)."""
    p = pool40_i.astype(np.float64)
    d = np.linalg.norm(p[:, None, :, :] - p[None, :, :, :], axis=-1)
    return np.maximum(d.mean(-1) / 0.10, d[:, :, -1] / 0.20)


def enumerate_legal(selected):
    rem = [j for j in selected if j < N_NAT]
    in_set = set(selected)
    add = [j for j in range(N_NAT, N_NAT + N_KIN) if j not in in_set]
    return [(r, a) for r in rem for a in add]


def score_pair(r, l, selected, pool40_i, D40):
    others = [x for x in selected if x != r]
    if not others:
        min_D = 0.0
        mean_ep = np.zeros(2, dtype=np.float64)
    else:
        o = np.asarray(others, dtype=np.int64)
        min_D = float(D40[l, o].min())
        mean_ep = pool40_i[o, -1, :].astype(np.float64).mean(0)
    ep_l = pool40_i[l, -1, :].astype(np.float64)
    tie_mean = -float(np.linalg.norm(ep_l - mean_ep))
    return (min_D, tie_mean, -int(r), -int(l))


def select_one(pool40_i, native_i, motion_stats, motion_gate):
    """Deploy: motion PASS + GT-free score. No GT.
    Semantics: argmax score among motion-PASS pairs.
    Impl: sort by score desc, first PASS == best among PASS.
    """
    selected = list(range(N_NAT))
    trace = []
    rejects = dict(motion=0, no_candidate=0)
    native_motion = motion_stats(native_i[None])
    D40 = pairwise_D_matrix(pool40_i)

    for step in range(MAX_SWAPS):
        pairs = enumerate_legal(selected)
        if not pairs:
            rejects["no_candidate"] += 1
            break
        scored = [(score_pair(r, a, selected, pool40_i, D40), r, a) for r, a in pairs]
        scored.sort(key=lambda t: t[0], reverse=True)
        chosen = None
        motion_fail = 0
        for key, r, a in scored:
            new_sel = [a if x == r else x for x in selected]
            paths = pool40_i[np.asarray(new_sel, dtype=np.int64)]
            mg = motion_gate(motion_stats(paths[None]), native_motion)
            if mg["status"] != "PASS":
                motion_fail += 1
                continue
            chosen = (r, a, mg, key)
            break
        rejects["motion"] += motion_fail
        if chosen is None:
            rejects["no_candidate"] += 1
            break
        r, a, mg, key = chosen
        selected = [a if x == r else x for x in selected]
        trace.append(dict(
            step=step,
            remove_native=int(r),
            add_kin=int(a),
            score_min_D=float(key[0]),
            score_neg_ep_to_mean=float(key[1]),
            motion_ratios=mg["ratios"],
            motion_fail_before_accept=int(motion_fail),
        ))
    idx = np.asarray(selected, dtype=np.int64)
    assert len(set(idx.tolist())) == K_SELECT
    return idx, trace, rejects


def select_batch(pool40, native, motion_stats, motion_gate, progress_every=100):
    B = pool40.shape[0]
    indices = np.zeros((B, K_SELECT), dtype=np.int64)
    traces = []
    reject_tot = dict(motion=0, no_candidate=0)
    for b in range(B):
        idx, tr, rej = select_one(pool40[b], native[b], motion_stats, motion_gate)
        indices[b] = idx
        traces.append(tr)
        for k in reject_tot:
            reject_tot[k] += rej[k]
        if progress_every and (b + 1) % progress_every == 0:
            print(
                f"    GTF {b+1}/{B} swaps_mean={np.mean([len(t) for t in traces]):.2f}",
                flush=True,
            )
    return indices, traces, reject_tot
