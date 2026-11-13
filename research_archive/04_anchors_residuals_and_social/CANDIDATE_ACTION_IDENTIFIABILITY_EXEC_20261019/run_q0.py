"""Main Execution Orchestrator for Q0 Action Identifiability Diagnostic.
Strictly implements the evaluation of 12 units (4 cohorts x 3 seeds),
records state trajectories S0..S3 with vectorized batch operations,
computes posthoc GT labels, and runs 10,000 bootstrap draws.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

EXEC_DIR = Path(__file__).resolve().parent
ROOT = EXEC_DIR.parent

from .frozen_adapter import (
    load_frozen_models, build_pool_features_22, pairwise_D_matrix,
    N_NAT, N_KIN,
)
from .kinematics import precompute_pool_motion, subset_motion_stats
from .enumerate_actions import observe_and_decide
from .posthoc_labels import candidate_metrics, set_metrics, batch_evaluate_action_labels
from .statistics import compute_window_stats, run_bootstrap_family8


def run_q0_diagnostic():
    print("=" * 70)
    print("STARTING Q0 ACTION IDENTIFIABILITY DIAGNOSTIC EXECUTION (HIGH-PERFORMANCE)")
    print("=" * 70)
    t_start = time.perf_counter()

    # Load data
    cache_path = ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz"
    encoder_path = ROOT / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz"
    labels_path = ROOT / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz"
    splits_path = ROOT / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz"

    with np.load(cache_path, allow_pickle=False) as z:
        all_native = z["candidate"][..., :24].reshape(-1, 20, 12, 2).astype(np.float64)
        all_sids = z["sample_id"].copy()
    with np.load(encoder_path, allow_pickle=False) as z:
        all_past = z["past"].astype(np.float64)
    with np.load(labels_path, allow_pickle=False) as z:
        all_gt = z["future"].astype(np.float64)
    with np.load(splits_path, allow_pickle=False) as z:
        sp_fit = z["grouped_train_index"]

    fit_gidx_to_local = {int(g): l for l, g in enumerate(sp_fit)}

    cohorts_npz = np.load(EXEC_DIR / "COHORT_IDS.npz", allow_pickle=True)
    cohort_names = ["C_seen", "C_same", "C_calA", "C_calB"]
    seeds = [42, 43, 44]

    # Output directories
    states_root = EXEC_DIR / "states"
    states_root.mkdir(parents=True, exist_ok=True)

    # To accumulate window data for statistics
    cohort_window_data = {c_name: {s: [] for s in seeds} for c_name in cohort_names}
    cohort_tracks_map = {c_name: list(cohorts_npz[f"{c_name}_tracks"]) for c_name in cohort_names}

    state_counts = {c_name: {s: {d: 0 for d in range(4)} for s in seeds} for c_name in cohort_names}
    total_actions_evaluated = 0

    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        models = load_frozen_models(s)
        kin_path = ROOT / f"UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_FIT_seed{s}.npy"
        kin_fit = np.load(kin_path, mmap_mode="r")

        # Load 32-pair training lookup for seed s
        rows_path = ROOT / f"CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k1_exec/seed{s}/train_data/rows_seed{s}.npz"
        rows_z = np.load(rows_path, allow_pickle=True)
        s0_mask = (rows_z["state"] == 0)
        seen_32_set = set(zip(
            rows_z["sample_id"][s0_mask],
            rows_z["rem"][s0_mask],
            rows_z["add"][s0_mask]
        ))

        for c_name in cohort_names:
            c_dir = states_root / f"seed_{s}" / c_name
            c_dir.mkdir(parents=True, exist_ok=True)

            idx_arr = cohorts_npz[f"{c_name}_idx"]
            sid_arr = cohorts_npz[f"{c_name}_ids"]
            trk_arr = cohorts_npz[f"{c_name}_tracks"]
            n_windows = len(idx_arr)

            t0_cohort = time.perf_counter()
            for w in range(n_windows):
                g_idx = int(idx_arr[w])
                sid = int(sid_arr[w])
                tk = str(trk_arr[w])
                fit_local = fit_gidx_to_local[g_idx]

                native_i = all_native[g_idx]
                kin_i = kin_fit[fit_local].astype(np.float64)
                pool40_i = np.concatenate([native_i, kin_i], axis=0)
                past_i = all_past[g_idx]
                gt_i = all_gt[g_idx]

                A_c, F_c, J_c = candidate_metrics(pool40_i, gt_i)
                _, _, j_nat, _ = set_metrics(A_c, F_c, J_c, list(range(20)))
                c_nat = int(j_nat <= 1.0)

                feat40_i = build_pool_features_22(past_i[None, ...], native_i[None, ...], kin_i[None, ...])[0]
                D40_i = pairwise_D_matrix(pool40_i)
                pm_cache = precompute_pool_motion(pool40_i)
                nat_stats = subset_motion_stats(pm_cache[0], pm_cache[1], pm_cache[2], list(range(20)))

                selected = list(range(20))
                win_arrays = {
                    "sample_id": np.array(sid, dtype=np.int64),
                    "track_key": np.array(tk, dtype=object),
                    "c_native": np.array(c_nat, dtype=np.int32),
                }

                depth_count = 0
                for depth in range(4):
                    state_counts[c_name][s][depth] += 1
                    depth_count += 1
                    obs = observe_and_decide(
                        pool40_i, native_i, feat40_i, D40_i, selected, models,
                        pool_motion_cache=pm_cache, native_stats=nat_stats
                    )
                    pairs = obs["pairs"]
                    n_pairs = len(pairs)
                    total_actions_evaluated += n_pairs

                    # Vectorized batch evaluation of labels
                    m_pass = obs["motion_pass"]
                    batch_lbls = batch_evaluate_action_labels(
                        A_c, F_c, J_c, selected, pairs, m_pass, c_nat
                    )

                    # Original training pair flag (only for C_seen at depth 0)
                    orig_trained_flags = np.array([
                        bool((c_name == "C_seen") and (depth == 0) and ((sid, r, a) in seen_32_set))
                        for r, a in pairs
                    ], dtype=bool)

                    # Record statistics for S0
                    if depth == 0:
                        q_pass = obs["diagnostic_q"][m_pass]
                        b_pass = batch_lbls["is_b"][m_pass]

                        ro_act = obs["rank_only_action"]
                        ro_is_b = False
                        if ro_act is not None:
                            ro_p_idx = ro_act[0]
                            ro_is_b = bool(batch_lbls["is_b"][ro_p_idx])

                        orig_act = obs["original_action"]
                        orig_caught_b = False
                        if orig_act is not None:
                            orig_p_idx = orig_act[0]
                            orig_caught_b = bool(batch_lbls["is_b"][orig_p_idx])

                        cohort_window_data[c_name][s].append({
                            "feasible_scores": q_pass,
                            "feasible_is_b": b_pass,
                            "rank_only_is_b": ro_is_b,
                            "opportunity_exists": bool(np.any(b_pass)),
                            "original_caught_b": orig_caught_b,
                        })

                    # Package flat arrays for this depth
                    orig_act_arr = np.array(obs["original_action"] if obs["original_action"] is not None else [-1, -1, -1], dtype=np.int32)
                    ro_act_arr = np.array(obs["rank_only_action"] if obs["rank_only_action"] is not None else [-1, -1, -1], dtype=np.int32)
                    deltas = np.column_stack([batch_lbls["dA"], batch_lbls["dF"], batch_lbls["dJ"], batch_lbls["dC"]])

                    win_arrays[f"d{depth}_selected"] = np.array(selected, dtype=np.int32)
                    win_arrays[f"d{depth}_pairs"] = np.array(pairs, dtype=np.int32)
                    win_arrays[f"d{depth}_preds"] = obs["pred"].astype(np.float64)
                    win_arrays[f"d{depth}_old_gate"] = obs["old_gate"]
                    win_arrays[f"d{depth}_motion_pass"] = obs["motion_pass"]
                    win_arrays[f"d{depth}_diagnostic_q"] = obs["diagnostic_q"].astype(np.float64)
                    win_arrays[f"d{depth}_is_b"] = batch_lbls["is_b"]
                    win_arrays[f"d{depth}_is_h_inc"] = batch_lbls["is_h_inc"]
                    win_arrays[f"d{depth}_is_h_ret"] = batch_lbls["is_h_ret"]
                    win_arrays[f"d{depth}_is_h"] = batch_lbls["is_h"]
                    win_arrays[f"d{depth}_is_n"] = batch_lbls["is_n"]
                    win_arrays[f"d{depth}_deltas"] = deltas.astype(np.float64)
                    win_arrays[f"d{depth}_L_after"] = batch_lbls["L_after"].astype(np.int32)
                    win_arrays[f"d{depth}_orig_trained_flags"] = orig_trained_flags
                    win_arrays[f"d{depth}_original_action"] = orig_act_arr
                    win_arrays[f"d{depth}_rank_only_action"] = ro_act_arr

                    if obs["original_action"] is None or depth == 3:
                        break
                    rem, add = obs["original_action"][1], obs["original_action"][2]
                    selected = [add if x == rem else x for x in selected]

                win_arrays["n_states"] = np.array(depth_count, dtype=np.int32)

                # Save fast compressed numpy file
                win_file = c_dir / f"window_{sid}.npz"
                np.savez_compressed(win_file, **win_arrays)

            t1_cohort = time.perf_counter()
            print(f"  [{c_name}] Seed {s}: {n_windows} windows completed in {t1_cohort - t0_cohort:.2f}s")

    # Point metrics computation
    print("\n" + "=" * 70)
    print("COMPUTING POINT ESTIMATES & WINDOW-LEVEL METRICS")
    print("=" * 70)
    point_results = {c_name: {} for c_name in cohort_names}
    for c_name in cohort_names:
        for s in seeds:
            st = compute_window_stats(cohort_window_data[c_name][s], cohort_tracks_map[c_name])
            point_results[c_name][f"seed_{s}"] = st

        # Average across seeds
        avg_st = {}
        for metric_k in point_results[c_name]["seed_42"].keys():
            vals = [point_results[c_name][f"seed_{s}"][metric_k] for s in seeds]
            avg_st[metric_k] = float(np.mean(vals))
        point_results[c_name]["three_seed_mean"] = avg_st

    # Run Bootstrap Family 8
    print("\n" + "=" * 70)
    print("RUNNING 10,000 BOOTSTRAP DRAWS FOR FAMILY 8 CONFIDENCE INTERVALS")
    print("=" * 70)
    t0_boot = time.perf_counter()
    bootstrap_results = run_bootstrap_family8(cohort_window_data, cohort_tracks_map, n_draws=10000, seed=20261019)
    t1_boot = time.perf_counter()
    print(f"Bootstrap finished in {t1_boot - t0_boot:.2f}s")

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
        "state_depth_counts": state_counts,
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

    print("\n" + "=" * 70)
    print(f"Q0 EXECUTION COMPLETE. BRANCH DECISION: {decision_branch}")
    print(f"Explanation: {decision_explanation}")
    print(f"Total time: {time.perf_counter() - t_start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_q0_diagnostic()
