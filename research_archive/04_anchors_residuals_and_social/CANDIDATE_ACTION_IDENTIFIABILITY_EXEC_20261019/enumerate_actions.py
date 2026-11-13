"""Action enumeration and state decision logic for Q0 Action Identifiability Diagnostic.
Strictly follows 01_Q0_实验设计.md Section 5 & 7.
Optimized with pool kinematic precomputation.
"""
from __future__ import annotations

import numpy as np
from .frozen_adapter import (
    batch_action_features, predict_actions,
    PRED_L_MAX, PRED_DA_MIN, PRED_DF_MIN, N_NAT, N_KIN,
)
from .kinematics import (
    precompute_pool_motion, subset_motion_stats, motion_gate, MOTION_FACTOR
)


def enumerate_legal_actions(selected: list | np.ndarray) -> list[tuple[int, int]]:
    """Enumerates legal swap actions: (remaining_native r, unused_KIN l)."""
    sel_set = set(selected)
    return [
        (int(r), int(l))
        for r in sorted(selected) if r < N_NAT
        for l in range(N_NAT, N_NAT + N_KIN) if l not in sel_set
    ]


def compute_diagnostic_q(dA: np.ndarray, dF: np.ndarray, dJ: np.ndarray, dC_c: np.ndarray, L_c: np.ndarray) -> np.ndarray:
    """Computes diagnostic margin score q = min((0.01 - L_c)/0.01, dA/0.2, dF/0.4, max(dC_c, dJ))."""
    term_L = (PRED_L_MAX - L_c) / PRED_L_MAX
    term_A = dA / 0.20
    term_F = dF / 0.40
    term_CJ = np.maximum(dC_c, dJ)
    return np.minimum(np.minimum(term_L, term_A), np.minimum(term_F, term_CJ))


def observe_and_decide(
    pool40: np.ndarray,
    native_paths: np.ndarray,
    feat40: np.ndarray,
    D40: np.ndarray,
    selected: list,
    models: dict,
    pool_motion_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    native_stats: dict[str, float] | None = None,
):
    """Enumerates all legal pairs, runs model predictions, kinematic gates,
    and determines original_action, rank_only_action, and diagnostic q.
    """
    pairs = enumerate_legal_actions(selected)
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {
            "pairs": [],
            "pred": np.zeros((0, 5), dtype=np.float64),
            "old_gate": np.zeros(0, dtype=bool),
            "motion_pass": np.zeros(0, dtype=bool),
            "diagnostic_q": np.zeros(0, dtype=np.float64),
            "original_action": None,
            "rank_only_action": None,
            "motion_details": [],
        }

    # 1. 113-d features & prediction
    X = batch_action_features(feat40, D40, selected, pairs)
    preds = predict_actions(models, X)
    dA = preds[:, 0]
    dF = preds[:, 1]
    dJ = preds[:, 2]
    dC = preds[:, 3]
    L = preds[:, 4]

    L_c = np.clip(L, 0.0, 1.0)
    dC_c = np.clip(dC, -1.0, 1.0)

    old_gate = (
        (L_c <= PRED_L_MAX)
        & (dA >= PRED_DA_MIN)
        & (dF >= PRED_DF_MIN)
        & ((dC_c > 0.0) | (dJ > 0.0))
    )

    diagnostic_q = compute_diagnostic_q(dA, dF, dJ, dC_c, L_c)

    # 2. Kinematic check for each pair using cached pool norms
    if pool_motion_cache is None:
        s40, a40, j40 = precompute_pool_motion(pool40)
    else:
        s40, a40, j40 = pool_motion_cache

    if native_stats is None:
        native_stats = subset_motion_stats(s40, a40, j40, list(range(N_NAT)))

    motion_pass = np.zeros(n_pairs, dtype=bool)
    motion_details = []
    for i, (rem, add) in enumerate(pairs):
        new_sel = [add if x == rem else x for x in selected]
        cand_stats = subset_motion_stats(s40, a40, j40, new_sel)
        mg = motion_gate(cand_stats, native_stats, factor=MOTION_FACTOR)
        is_pass = (mg["status"] == "PASS")
        motion_pass[i] = is_pass
        motion_details.append(mg)

    # 3. Lexicographic ranking
    # Key: (dC_c, dJ, dA + 0.5*dF, -rem, -add)
    # Higher is better
    ranked_indices = sorted(
        range(n_pairs),
        key=lambda i: (
            float(dC_c[i]),
            float(dJ[i]),
            float(dA[i] + 0.5 * dF[i]),
            -pairs[i][0],
            -pairs[i][1],
        ),
        reverse=True,
    )

    # 4. Action selection
    original_action = None
    for idx in ranked_indices:
        if motion_pass[idx] and old_gate[idx]:
            original_action = (idx, pairs[idx][0], pairs[idx][1])
            break

    rank_only_action = None
    for idx in ranked_indices:
        if motion_pass[idx]:
            rank_only_action = (idx, pairs[idx][0], pairs[idx][1])
            break

    return {
        "pairs": pairs,
        "pred": preds,
        "dA": dA,
        "dF": dF,
        "dJ": dJ,
        "dC": dC,
        "L": L,
        "dC_c": dC_c,
        "L_c": L_c,
        "old_gate": old_gate,
        "motion_pass": motion_pass,
        "diagnostic_q": diagnostic_q,
        "ranked_indices": ranked_indices,
        "original_action": original_action,
        "rank_only_action": rank_only_action,
        "motion_details": motion_details,
    }
