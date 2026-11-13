"""Posthoc Ground Truth Labels for Q0 Action Identifiability Diagnostic.
Strictly follows 01_Q0_实验设计.md Section 6.
Labels are computed strictly after decisions are frozen.
Includes vectorized batch evaluation.
"""
from __future__ import annotations

import numpy as np

QUALITY_A = 0.2
QUALITY_F = 0.4


def candidate_metrics(candidates: np.ndarray, gt: np.ndarray):
    """Computes A_j, F_j, J_j for all candidates.
    candidates: (K, 12, 2)
    gt: (12, 2)
    Returns: A (K,), F (K,), J (K,)
    """
    diff = candidates.astype(np.float64) - gt[None, :, :].astype(np.float64)
    dist = np.linalg.norm(diff, axis=-1)
    A = dist.mean(axis=-1)
    F = dist[:, -1]
    J = np.maximum(A / QUALITY_A, F / QUALITY_F)
    return A, F, J


def set_metrics(A: np.ndarray, F: np.ndarray, J: np.ndarray, selected: list | np.ndarray):
    """Computes a(S), f(S), j(S), c(S) for a given selected subset."""
    sel = np.asarray(selected, dtype=np.int64)
    a_S = float(np.min(A[sel]))
    f_S = float(np.min(F[sel]))
    j_S = float(np.min(J[sel]))
    c_S = int(j_S <= 1.0)
    return a_S, f_S, j_S, c_S


def evaluate_action_labels(
    A: np.ndarray,
    F: np.ndarray,
    J: np.ndarray,
    selected: list | np.ndarray,
    rem: int,
    add: int,
    motion_pass: bool,
    c_native: int,
):
    """Single action evaluation (used for fixtures and unit tests)."""
    a_before, f_before, j_before, c_before = set_metrics(A, F, J, selected)
    new_sel = [add if x == rem else x for x in selected]
    a_after, f_after, j_after, c_after = set_metrics(A, F, J, new_sel)

    dA = a_before - a_after
    dF = f_before - f_after
    dJ = j_before - j_after
    dC = c_after - c_before

    L_before = int(c_native * (1 - c_before))
    L_after = int(c_native * (1 - c_after))

    is_b = False
    is_h_inc = False
    is_h_ret = False
    is_h = False
    is_n = False
    harm_reasons = []

    if motion_pass:
        if (
            L_after == 0
            and dA >= 0.0
            and dF >= 0.0
            and dJ >= 0.0
            and dC >= 0
            and (dC > 0 or dJ > 0.0)
        ):
            is_b = True

        if dA < 0.0 or dF < 0.0 or dJ < 0.0 or dC < 0:
            is_h_inc = True
            if dA < 0.0:
                harm_reasons.append("dA_neg")
            if dF < 0.0:
                harm_reasons.append("dF_neg")
            if dJ < 0.0:
                harm_reasons.append("dJ_neg")
            if dC < 0:
                harm_reasons.append("dC_neg")

        if L_after == 1:
            is_h_ret = True
            harm_reasons.append("new_loss" if L_before == 0 else "persisting_loss")

        is_h = is_h_inc or is_h_ret
        if not is_b and not is_h:
            is_n = True

        assert (int(is_b) + int(is_h) + int(is_n)) == 1

    return {
        "dA": float(dA),
        "dF": float(dF),
        "dJ": float(dJ),
        "dC": int(dC),
        "L_before": int(L_before),
        "L_after": int(L_after),
        "is_b": bool(is_b),
        "is_h_inc": bool(is_h_inc),
        "is_h_ret": bool(is_h_ret),
        "is_h": bool(is_h),
        "is_n": bool(is_n),
        "harm_reasons": harm_reasons,
        "a_after": float(a_after),
        "f_after": float(f_after),
        "j_after": float(j_after),
        "c_after": int(c_after),
    }


def batch_evaluate_action_labels(
    A: np.ndarray,
    F: np.ndarray,
    J: np.ndarray,
    selected: list | np.ndarray,
    pairs: list[tuple[int, int]],
    motion_pass: np.ndarray,
    c_native: int,
):
    """Vectorized batch evaluation across all legal pairs in a state."""
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {
            "dA": np.zeros(0, dtype=np.float64),
            "dF": np.zeros(0, dtype=np.float64),
            "dJ": np.zeros(0, dtype=np.float64),
            "dC": np.zeros(0, dtype=np.int32),
            "L_after": np.zeros(0, dtype=np.int32),
            "is_b": np.zeros(0, dtype=bool),
            "is_h_inc": np.zeros(0, dtype=bool),
            "is_h_ret": np.zeros(0, dtype=bool),
            "is_h": np.zeros(0, dtype=bool),
            "is_n": np.zeros(0, dtype=bool),
        }

    sel = np.asarray(selected, dtype=np.int64)
    a_before = float(np.min(A[sel]))
    f_before = float(np.min(F[sel]))
    j_before = float(np.min(J[sel]))
    c_before = int(j_before <= 1.0)
    L_before = int(c_native * (1 - c_before))

    rem_arr = np.array([r for r, l in pairs], dtype=np.int64)
    add_arr = np.array([l for r, l in pairs], dtype=np.int64)

    # Precompute min without each rem in selected
    a_no_r = {r: float(np.min([A[s] for s in sel if s != r])) for r in sel}
    f_no_r = {r: float(np.min([F[s] for s in sel if s != r])) for r in sel}
    j_no_r = {r: float(np.min([J[s] for s in sel if s != r])) for r in sel}

    a_after = np.minimum(np.array([a_no_r[r] for r in rem_arr]), A[add_arr])
    f_after = np.minimum(np.array([f_no_r[r] for r in rem_arr]), F[add_arr])
    j_after = np.minimum(np.array([j_no_r[r] for r in rem_arr]), J[add_arr])
    c_after = (j_after <= 1.0).astype(np.int32)
    L_after = (c_native * (1 - c_after)).astype(np.int32)

    dA = a_before - a_after
    dF = f_before - f_after
    dJ = j_before - j_after
    dC = c_after - c_before

    # B: safe beneficial
    is_b = (
        motion_pass
        & (L_after == 0)
        & (dA >= 0.0)
        & (dF >= 0.0)
        & (dJ >= 0.0)
        & (dC >= 0)
        & ((dC > 0) | (dJ > 0.0))
    )

    # H_inc: increased error harm
    is_h_inc = motion_pass & (~is_b) & ((dA < 0.0) | (dF < 0.0) | (dJ < 0.0) | (dC < 0))

    # H_ret: native retention loss
    is_h_ret = motion_pass & (~is_b) & (L_after == 1)

    is_h = is_h_inc | is_h_ret
    is_n = motion_pass & (~is_b) & (~is_h)

    # Verification of mutual exclusivity
    assert np.all((is_b.astype(int) + is_h.astype(int) + is_n.astype(int))[motion_pass] == 1)

    return {
        "dA": dA,
        "dF": dF,
        "dJ": dJ,
        "dC": dC,
        "a_before": a_before,
        "f_before": f_before,
        "j_before": j_before,
        "c_before": c_before,
        "a_after": a_after,
        "f_after": f_after,
        "j_after": j_after,
        "c_after": c_after,
        "L_before": L_before,
        "L_after": L_after,
        "is_b": is_b,
        "is_h_inc": is_h_inc,
        "is_h_ret": is_h_ret,
        "is_h": is_h,
        "is_n": is_n,
    }
