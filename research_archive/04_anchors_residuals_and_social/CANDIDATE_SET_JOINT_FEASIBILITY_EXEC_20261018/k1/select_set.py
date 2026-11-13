"""Deploy I/S selectors. GT-free; same action space / motion gate as K0-C."""
from __future__ import annotations
import numpy as np
from .config import (
    N_NAT, K_SELECT, MAX_SWAPS, PRED_L_MAX, PRED_DA_MIN, PRED_DF_MIN,
)
from .features113 import action_feature_113, pairwise_D_matrix
from .sampling import enumerate_all_legal_actions
from .train_hgb import predict_arm

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from k0.metrics import motion_stats, motion_gate


def _accept_from_pred(pred):
    dA, dF, dJ, dC, L = pred.T
    L_c = np.clip(L, 0.0, 1.0)
    dC_c = np.clip(dC, -1.0, 1.0)
    ok = (
        (L_c <= PRED_L_MAX)
        & (dA >= PRED_DA_MIN)
        & (dF >= PRED_DF_MIN)
        & ((dC_c > 0) | (dJ > 0))
    )
    return ok, dA, dF, dJ, dC_c, L_c


def select_one(pool40_i, native_i, feat40_i, D40_i, models, native_motion=None):
    selected = list(range(N_NAT))
    trace = []
    rejects = dict(motion=0, pred_gate=0, no_candidate=0)
    if native_motion is None:
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
        ok, dA, dF, dJ, dC, L = _accept_from_pred(pred)
        cand_idx = np.where(ok)[0]
        if cand_idx.size == 0:
            rejects["pred_gate"] += len(pairs)
            break
        # motion-check only pred-qualified; among those, rank
        best = None
        best_key = None
        motion_fail = 0
        for ii in cand_idx:
            rem, add = pairs[ii]
            new_sel = [add if x == rem else x for x in selected]
            paths = pool40_i[np.array(new_sel, dtype=np.int64)]
            mg = motion_gate(motion_stats(paths[None]), native_motion)
            if mg["status"] != "PASS":
                motion_fail += 1
                continue
            key = (float(dC[ii]), float(dJ[ii]), float(dA[ii] + 0.5 * dF[ii]), -rem, -add)
            if best_key is None or key > best_key:
                best_key = key
                best = (ii, rem, add, mg)
        rejects["motion"] += motion_fail
        if best is None:
            if motion_fail:
                rejects["no_candidate"] += 1
            break
        ii, rem, add, mg = best
        selected = [add if x == rem else x for x in selected]
        trace.append(dict(
            step=step, remove_native=int(rem), add_kin=int(add),
            pred_dA=float(dA[ii]), pred_dF=float(dF[ii]), pred_dJ=float(dJ[ii]),
            pred_dC=float(dC[ii]), pred_L=float(L[ii]),
            motion_ratios=mg["ratios"],
        ))
    idx = np.array(selected, dtype=np.int64)
    assert len(set(idx.tolist())) == K_SELECT
    return idx, trace, rejects


def select_batch(pool40, native, feat40, models, D_cache=None, progress_every=50):
    B = pool40.shape[0]
    indices = np.zeros((B, K_SELECT), dtype=np.int64)
    traces = []
    reject_tot = dict(motion=0, pred_gate=0, no_candidate=0)
    for b in range(B):
        D40 = D_cache[b] if D_cache is not None else pairwise_D_matrix(pool40[b])
        nm = motion_stats(native[b:b+1])
        idx, tr, rej = select_one(pool40[b], native[b], feat40[b], D40, models, native_motion=nm)
        indices[b] = idx
        traces.append(tr)
        for k in reject_tot:
            reject_tot[k] += rej[k]
        if progress_every and (b + 1) % progress_every == 0:
            print(f"    select {b+1}/{B} swaps_mean={np.mean([len(t) for t in traces]):.2f}", flush=True)
    return indices, traces, reject_tot
