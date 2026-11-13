"""Finish Q0 Evaluation by loading already completed state files from disk.
Computes point estimates, runs fast 10,000 bootstrap, determines branch decision,
executes audit_saved.py, and generates full reports.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

EXEC_DIR = Path(__file__).resolve().parent
ROOT = EXEC_DIR.parent

from .statistics import compute_window_stats, run_fast_bootstrap_family8
from .audit_saved import run_saved_audit
from .report import generate_reports


def finish():
    print("=" * 70)
    print("FINISHING Q0 EVALUATION FROM SAVED STATE ARRAYS")
    print("=" * 70)
    t_start = time.perf_counter()

    cohorts_npz = np.load(EXEC_DIR / "COHORT_IDS.npz", allow_pickle=True)
    cohort_names = ["C_seen", "C_same", "C_calA", "C_calB"]
    seeds = [42, 43, 44]

    states_root = EXEC_DIR / "states"
    assert states_root.exists(), "states directory does not exist!"

    cohort_window_data = {c_name: {s: [] for s in seeds} for c_name in cohort_names}
    cohort_tracks_map = {c_name: list(cohorts_npz[f"{c_name}_tracks"]) for c_name in cohort_names}

    state_depth_counts = {c_name: {s: {d: 0 for d in range(4)} for s in seeds} for c_name in cohort_names}
    total_actions_evaluated = 0

    print("Loading saved window states...")
    for s in seeds:
        for c_name in cohort_names:
            c_dir = states_root / f"seed_{s}" / c_name
            sid_arr = cohorts_npz[f"{c_name}_ids"]
            for sid in sid_arr:
                wf = c_dir / f"window_{sid}.npz"
                assert wf.exists(), f"Missing window file: {wf}"
                data = np.load(wf, allow_pickle=True)
                n_states = int(data["n_states"])

                for d in range(n_states):
                    state_depth_counts[c_name][s][d] += 1
                    total_actions_evaluated += len(data[f"d{d}_pairs"])

                # Depth 0 record for statistics
                m_pass = data["d0_motion_pass"]
                q_pass = data["d0_diagnostic_q"][m_pass]
                b_pass = data["d0_is_b"][m_pass]

                ro_act = data["d0_rank_only_action"]
                ro_is_b = False
                if ro_act[0] != -1:
                    ro_is_b = bool(data["d0_is_b"][ro_act[0]])

                orig_act = data["d0_original_action"]
                orig_caught_b = False
                if orig_act[0] != -1:
                    orig_caught_b = bool(data["d0_is_b"][orig_act[0]])

                cohort_window_data[c_name][s].append({
                    "feasible_scores": q_pass,
                    "feasible_is_b": b_pass,
                    "rank_only_is_b": ro_is_b,
                    "opportunity_exists": bool(np.any(b_pass)),
                    "original_caught_b": orig_caught_b,
                })

    print(f"Loaded all 1536 windows. Total actions evaluated: {total_actions_evaluated}")

    # Point metrics computation
    print("\nComputing point estimates & window-level metrics...")
    point_results = {c_name: {} for c_name in cohort_names}
    for c_name in cohort_names:
        for s in seeds:
            st = compute_window_stats(cohort_window_data[c_name][s], cohort_tracks_map[c_name])
            point_results[c_name][f"seed_{s}"] = st

        avg_st = {}
        for metric_k in point_results[c_name]["seed_42"].keys():
            vals = [point_results[c_name][f"seed_{s}"][metric_k] for s in seeds]
            avg_st[metric_k] = float(np.mean(vals))
        point_results[c_name]["three_seed_mean"] = avg_st

    # Run Fast Bootstrap Family 8
    print("\nRunning fast 10,000 bootstrap draws for Family 8...")
    t0_boot = time.perf_counter()
    bootstrap_results = run_fast_bootstrap_family8(cohort_window_data, cohort_tracks_map, n_draws=10000, seed=20261019)
    t1_boot = time.perf_counter()
    print(f"Bootstrap finished in {t1_boot - t0_boot:.2f}s!")

    # Evaluate RANK_SIGNAL for each cohort
    rank_signal = {}
    for c_name in cohort_names:
        boot_c = bootstrap_results[c_name]
        valid_dr = boot_c["valid_draws"] >= 9500

        all_pos_tracks = all(
            point_results[c_name][f"seed_{s}"]["positive_tracks_count"] >= 20
            for s in seeds
        )

        ap_bonf_pos = (boot_c["ap_diff"]["bonferroni_ci"][0] > 0.0)
        rank_bonf_pos = (boot_c["rank_diff"]["bonferroni_ci"][0] > 0.0)

        ap_point_pos = all(point_results[c_name][f"seed_{s}"]["ap_minus_base"] > 0.0 for s in seeds)
        rank_point_pos = all(point_results[c_name][f"seed_{s}"]["rank_diff"] > 0.0 for s in seeds)

        has_signal = bool(valid_dr and all_pos_tracks and ap_bonf_pos and rank_bonf_pos and ap_point_pos and rank_point_pos)
        rank_signal[c_name] = {
            "has_rank_signal": has_signal,
            "valid_draws_ok": valid_dr,
            "min_positive_tracks_ok": all_pos_tracks,
            "ap_bonferroni_positive": ap_bonf_pos,
            "rank_bonferroni_positive": rank_bonf_pos,
            "ap_point_positive": ap_point_pos,
            "rank_point_positive": rank_point_pos,
            "min_pos_tracks_by_seed": [point_results[c_name][f"seed_{s}"]["positive_tracks_count"] for s in seeds],
        }

    # Branch Decision
    c_calA_sig = rank_signal["C_calA"]["has_rank_signal"]
    c_calB_sig = rank_signal["C_calB"]["has_rank_signal"]
    c_seen_sig = rank_signal["C_seen"]["has_rank_signal"]
    c_same_sig = rank_signal["C_same"]["has_rank_signal"]

    if c_calA_sig and c_calB_sig:
        decision_branch = "CALIBRATION_BRANCH"
        decision_explanation = "C_calA and C_calB both exhibit significant RANK_SIGNAL. Ranking signal transfers across tracks; proceed to accept/abstain threshold calibration."
    elif c_seen_sig and not c_same_sig and not (c_calA_sig or c_calB_sig):
        decision_branch = "TRAINING_EXPOSURE_OVERFIT"
        decision_explanation = "C_seen has signal, but C_same and calibration cohorts lack signal. Strong training exposure dependency; inspect fitting mechanisms."
    elif (c_seen_sig or c_same_sig) and not (c_calA_sig or c_calB_sig):
        decision_branch = "TRACK_TRANSFER_GAP"
        decision_explanation = "Seen/same tracks show ranking signal, but held-out calibration tracks lack signal. Representation fails to generalize to new tracks."
    elif not any(rank_signal[c]["has_rank_signal"] for c in cohort_names):
        decision_branch = "DISCRIMINATION_DEFICIT"
        decision_explanation = "No cohort shows significant RANK_SIGNAL. The frozen regression scores lack discriminative power between beneficial and non-beneficial actions."
    else:
        decision_branch = "MIXED_OR_INSUFFICIENT_EVIDENCE"
        decision_explanation = "Signals are mixed or confidence intervals cross zero without clear resolution."

    # Save outputs
    metrics_summary = {
        "tag": "q0_action_identifiability_metrics",
        "total_actions_evaluated": total_actions_evaluated,
        "state_depth_counts": state_depth_counts,
        "point_estimates": point_results,
        "bootstrap_family8": bootstrap_results,
        "rank_signals": rank_signal,
    }

    branch_decision = {
        "tag": "q0_branch_decision",
        "branch": decision_branch,
        "explanation": decision_explanation,
        "rank_signals": rank_signal,
        "execution_wall_seconds": time.perf_counter() - t_start,
    }

    with (EXEC_DIR / "STATE_METRICS.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with (EXEC_DIR / "BOOTSTRAP_FAMILY8.json").open("w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)

    with (EXEC_DIR / "BRANCH_DECISION.json").open("w", encoding="utf-8") as f:
        json.dump(branch_decision, f, indent=2)

    print(f"\nBranch Decision: {decision_branch}")
    print(f"Explanation: {decision_explanation}")

    # Run independent audit
    print("\nRunning independent saved state audit...")
    run_saved_audit()

    # Generate reports
    print("\nGenerating final markdown and json reports...")
    generate_reports()

    print("\n" + "=" * 70)
    print(f"EVALUATION FULLY COMPLETE IN {time.perf_counter() - t_start:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    finish()
