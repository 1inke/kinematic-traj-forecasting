"""Motion-only gate + HGB rank. No pred_gate. Same action space / rank key as K1."""
from __future__ import annotations
import numpy as np

N_NAT, N_KIN, K_SELECT, MAX_SWAPS = 20, 20, 20, 20


def select_one(pool40_i, native_i, feat40_i, D40_i, models, action_feature_113,
               enumerate_all_legal_actions, predict_arm, motion_stats, motion_gate):
    selected = list(range(N_NAT))
    trace = []
    rejects = dict(motion=0, no_candidate=0, n_scored=0)
    native_motion = motion_stats(native_i[None])

    for step in range(MAX_SWAPS):
        pairs = enumerate_all_legal_actions(selected)
        if not pairs:
            rejects["no_candidate"] += 1
            break
        X = np.stack([
            action_feature_113(feat40_i, D40_i, selected, rem, add)
            for rem, add in pairs
        ], 0)
        pred = predict_arm(models, X)
        # OUTPUT_NAMES order: dA, dF, dJ, dC, L
        dA, dF, dJ, dC, L = pred.T
        dC_c = np.clip(dC, -1.0, 1.0)
        L_c = np.clip(L, 0.0, 1.0)
        rejects["n_scored"] += len(pairs)

        best = None
        best_key = None
        motion_fail = 0
        for ii, (rem, add) in enumerate(pairs):
            new_sel = [add if x == rem else x for x in selected]
            paths = pool40_i[np.array(new_sel, dtype=np.int64)]
            mg = motion_gate(motion_stats(paths[None]), native_motion)
            if mg["status"] != "PASS":
                motion_fail += 1
                continue
            key = (float(dC_c[ii]), float(dJ[ii]), float(dA[ii] + 0.5 * dF[ii]), -rem, -add)
            if best_key is None or key > best_key:
                best_key = key
                best = (ii, rem, add, mg)
        rejects["motion"] += motion_fail
        if best is None:
            rejects["no_candidate"] += 1
            break
        ii, rem, add, mg = best
        selected = [add if x == rem else x for x in selected]
        trace.append(dict(
            step=step, remove_native=int(rem), add_kin=int(add),
            pred_dA=float(dA[ii]), pred_dF=float(dF[ii]), pred_dJ=float(dJ[ii]),
            pred_dC=float(dC_c[ii]), pred_L=float(L_c[ii]),
            motion_ratios=mg["ratios"],
        ))
    idx = np.array(selected, dtype=np.int64)
    assert len(set(idx.tolist())) == K_SELECT
    return idx, trace, rejects


def select_batch(pool40, native, feat40, models, D_cache,
                 action_feature_113, enumerate_all_legal_actions, predict_arm,
                 pairwise_D_matrix, motion_stats, motion_gate, progress_every=100, tag="B"):
    B = pool40.shape[0]
    indices = np.zeros((B, K_SELECT), dtype=np.int64)
    traces = []
    reject_tot = dict(motion=0, no_candidate=0, n_scored=0)
    for b in range(B):
        D40 = D_cache[b] if D_cache is not None else pairwise_D_matrix(pool40[b])
        idx, tr, rej = select_one(
            pool40[b], native[b], feat40[b], D40, models,
            action_feature_113, enumerate_all_legal_actions, predict_arm,
            motion_stats, motion_gate,
        )
        indices[b] = idx
        traces.append(tr)
        for k in reject_tot:
            reject_tot[k] += rej[k]
        if progress_every and (b + 1) % progress_every == 0:
            print(
                f"    {tag} {b+1}/{B} swaps_mean={np.mean([len(t) for t in traces]):.2f}",
                flush=True,
            )
    return indices, traces, reject_tot
