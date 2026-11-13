"""Independent Saved State Audit for Q0 Action Identifiability Diagnostic.
Strictly implements 03_实现接口与验收.md Section 6.
Does NOT call models or predict functions; reads saved .npz files directly and verifies arithmetic consistency.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

EXEC_DIR = Path(__file__).resolve().parent


def run_saved_audit() -> dict:
    print("=" * 70)
    print("RUNNING INDEPENDENT AUDIT ON SAVED NPZ FILES")
    print("=" * 70)

    states_root = EXEC_DIR / "states"
    assert states_root.exists(), "states directory does not exist!"

    cohort_names = ["C_seen", "C_same", "C_calA", "C_calB"]
    seeds = [42, 43, 44]

    audit_summary = {
        "status": "PASS",
        "units_audited": 0,
        "total_windows_audited": 0,
        "total_states_audited": 0,
        "total_actions_audited": 0,
        "arithmetic_discrepancies": 0,
        "action_selection_discrepancies": 0,
    }

    for s in seeds:
        for c_name in cohort_names:
            c_dir = states_root / f"seed_{s}" / c_name
            win_files = list(c_dir.glob("window_*.npz"))
            assert len(win_files) == 128, f"Expected 128 window files for seed {s} {c_name}, found {len(win_files)}"

            audit_summary["units_audited"] += 1
            audit_summary["total_windows_audited"] += len(win_files)

            for wf in win_files:
                data = np.load(wf, allow_pickle=True)
                n_states = int(data["n_states"])
                audit_summary["total_states_audited"] += n_states

                for d in range(n_states):
                    pairs = data[f"d{d}_pairs"]
                    preds = data[f"d{d}_preds"]
                    old_gate = data[f"d{d}_old_gate"]
                    motion_pass = data[f"d{d}_motion_pass"]
                    is_b = data[f"d{d}_is_b"]
                    is_h = data[f"d{d}_is_h"]
                    is_n = data[f"d{d}_is_n"]
                    orig_act = data[f"d{d}_original_action"]

                    n_pairs = len(pairs)
                    audit_summary["total_actions_audited"] += n_pairs

                    # Verify label mutual exclusivity within motion_pass
                    sum_labels = is_b.astype(int) + is_h.astype(int) + is_n.astype(int)
                    if not np.all(sum_labels[motion_pass] == 1):
                        audit_summary["arithmetic_discrepancies"] += 1

                    # Verify old gate logic
                    dA = preds[:, 0]
                    dF = preds[:, 1]
                    dJ = preds[:, 2]
                    dC = preds[:, 3]
                    L = preds[:, 4]
                    dC_c = np.clip(dC, -1.0, 1.0)
                    L_c = np.clip(L, 0.0, 1.0)

                    recheck_old_gate = (
                        (L_c <= 0.01)
                        & (dA >= 0.0)
                        & (dF >= 0.0)
                        & ((dC_c > 0.0) | (dJ > 0.0))
                    )
                    if not np.array_equal(old_gate, recheck_old_gate):
                        audit_summary["action_selection_discrepancies"] += 1

                    # Verify ranking and action selection
                    ranked_indices = sorted(
                        range(n_pairs),
                        key=lambda i: (
                            float(dC_c[i]),
                            float(dJ[i]),
                            float(dA[i] + 0.5 * dF[i]),
                            -int(pairs[i, 0]),
                            -int(pairs[i, 1]),
                        ),
                        reverse=True,
                    )

                    expected_orig = None
                    for idx in ranked_indices:
                        if motion_pass[idx] and old_gate[idx]:
                            expected_orig = (idx, int(pairs[idx, 0]), int(pairs[idx, 1]))
                            break

                    saved_orig = tuple(orig_act.tolist()) if orig_act[0] != -1 else None
                    if saved_orig != expected_orig:
                        audit_summary["action_selection_discrepancies"] += 1

    if (
        audit_summary["arithmetic_discrepancies"] > 0
        or audit_summary["action_selection_discrepancies"] > 0
    ):
        audit_summary["status"] = "FAIL"

    with (EXEC_DIR / "SAVED_AUDIT.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"Audit completed: {audit_summary['total_windows_audited']} windows across {audit_summary['units_audited']} units.")
    print(f"Total actions: {audit_summary['total_actions_audited']}, Discrepancies: {audit_summary['arithmetic_discrepancies'] + audit_summary['action_selection_discrepancies']}")
    print(f"Overall Audit Status: {audit_summary['status']}")
    return audit_summary


if __name__ == "__main__":
    run_saved_audit()
