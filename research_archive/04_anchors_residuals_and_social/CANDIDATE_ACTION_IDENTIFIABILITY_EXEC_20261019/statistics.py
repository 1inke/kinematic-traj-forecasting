"""Statistical evaluation, Weighted AP, and Fast Bootstrap CI for Q0 Action Identifiability Diagnostic.
Strictly follows 02_评价与后续分支.md Section 2 & 3.
Implements group precomputation and vectorized group accumulation for high performance.
"""
from __future__ import annotations

import numpy as np


def compute_weighted_ap(scores: np.ndarray, is_b: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Computes weighted Average Precision with grouped ties.
    Returns: (AP, base_rate)
    """
    total_w = float(np.sum(weights))
    total_b_w = float(np.sum(weights[is_b]))
    if total_w <= 0.0 or total_b_w <= 0.0:
        return 0.0, 0.0

    base_rate = total_b_w / total_w

    sort_idx = np.argsort(-scores)
    sorted_scores = scores[sort_idx]
    sorted_is_b = is_b[sort_idx]
    sorted_weights = weights[sort_idx]

    unique_scores, group_indices = np.unique(sorted_scores, return_inverse=True)
    n_groups = len(unique_scores)

    ap = 0.0
    cum_w = 0.0
    cum_b_w = 0.0

    for g_idx in range(n_groups - 1, -1, -1):
        mask = (group_indices == g_idx)
        g_w = np.sum(sorted_weights[mask])
        g_b_w = np.sum(sorted_weights[mask & sorted_is_b])

        cum_w += g_w
        cum_b_w += g_b_w

        precision = cum_b_w / cum_w
        delta_recall = g_b_w / total_b_w
        ap += delta_recall * precision

    return float(ap), float(base_rate)


def compute_window_stats(window_records: list[dict], track_keys: list[str]) -> dict:
    """Computes point estimates for a single cohort and single seed at depth S0."""
    n_win = len(window_records)
    assert n_win == len(track_keys)
    if n_win == 0:
        return {}

    unique_tracks, track_inv, track_counts = np.unique(track_keys, return_inverse=True, return_counts=True)
    n_tracks = len(unique_tracks)
    track_weight = 1.0 / n_tracks

    all_scores = []
    all_is_b = []
    all_act_weights = []

    win_weights = np.zeros(n_win, dtype=np.float64)
    rank_only_flags = np.zeros(n_win, dtype=np.float64)
    uniform_expecteds = np.zeros(n_win, dtype=np.float64)
    opp_exists_flags = np.zeros(n_win, dtype=np.float64)
    opp_caught_flags = np.zeros(n_win, dtype=np.float64)

    tracks_with_b = set()

    for i, (rec, tk) in enumerate(zip(window_records, track_keys)):
        t_idx = track_inv[i]
        w_win = track_weight / track_counts[t_idx]
        win_weights[i] = w_win

        scores_i = rec["feasible_scores"]
        is_b_i = rec["feasible_is_b"]
        n_pass = len(scores_i)

        if n_pass > 0:
            w_act = w_win / n_pass
            all_scores.append(scores_i)
            all_is_b.append(is_b_i)
            all_act_weights.append(np.full(n_pass, w_act, dtype=np.float64))
            b_count = int(np.sum(is_b_i))
            uniform_expecteds[i] = b_count / n_pass
            if b_count > 0:
                tracks_with_b.add(tk)

        rank_only_flags[i] = 1.0 if rec["rank_only_is_b"] else 0.0
        opp_exists_flags[i] = 1.0 if rec["opportunity_exists"] else 0.0
        opp_caught_flags[i] = 1.0 if rec["original_caught_b"] else 0.0

    if all_scores:
        cat_scores = np.concatenate(all_scores)
        cat_is_b = np.concatenate(all_is_b)
        cat_weights = np.concatenate(all_act_weights)
        ap, base_rate = compute_weighted_ap(cat_scores, cat_is_b, cat_weights)
    else:
        ap, base_rate = 0.0, 0.0

    norm_win_w = win_weights / np.sum(win_weights)
    rank_only_rate = float(np.sum(norm_win_w * rank_only_flags))
    uniform_rate = float(np.sum(norm_win_w * uniform_expecteds))
    rank_diff = rank_only_rate - uniform_rate

    opp_rate = float(np.sum(norm_win_w * opp_exists_flags))
    opp_weight_sum = float(np.sum(norm_win_w * opp_exists_flags))
    if opp_weight_sum > 0:
        opp_catch_rate = float(np.sum(norm_win_w * opp_caught_flags) / opp_weight_sum)
    else:
        opp_catch_rate = 0.0

    return {
        "ap": ap,
        "base_rate": base_rate,
        "ap_minus_base": ap - base_rate,
        "rank_only_rate": rank_only_rate,
        "uniform_expected_rate": uniform_rate,
        "rank_diff": rank_diff,
        "opportunity_rate": opp_rate,
        "opportunity_catch_rate": opp_catch_rate,
        "positive_tracks_count": len(tracks_with_b),
        "total_tracks_count": n_tracks,
        "total_windows_count": n_win,
    }


def run_fast_bootstrap_family8(
    cohort_data: dict[str, dict[int, list[dict]]],
    cohort_tracks: dict[str, list[str]],
    n_draws: int = 10000,
    seed: int = 20261019,
) -> dict:
    """Vectorized 10,000 bootstrap draws with group precomputation."""
    rng = np.random.default_rng(seed)
    cohort_names = ["C_seen", "C_same", "C_calA", "C_calB"]
    seeds = [42, 43, 44]

    # Pre-structure window track arrays
    cohort_meta = {}
    precomputed = {}

    for c_name in cohort_names:
        t_keys = np.array(cohort_tracks[c_name], dtype=object)
        u_tracks, inv = np.unique(t_keys, return_inverse=True)
        n_tr = len(u_tracks)
        n_win = len(t_keys)
        cohort_meta[c_name] = {
            "u_tracks": u_tracks,
            "inv": inv,
            "n_tracks": n_tr,
            "n_windows": n_win,
        }

        # Precompute per seed
        precomputed[c_name] = {}
        for s in seeds:
            win_recs = cohort_data[c_name][s]
            scores_list = []
            is_b_list = []
            win_idx_list = []
            win_n_pass = np.zeros(n_win, dtype=np.float64)
            win_rank_only_is_b = np.zeros(n_win, dtype=np.float64)
            win_uniform_exp = np.zeros(n_win, dtype=np.float64)

            for w_idx, rec in enumerate(win_recs):
                sc = rec["feasible_scores"]
                ib = rec["feasible_is_b"]
                n_p = len(sc)
                win_n_pass[w_idx] = n_p
                if n_p > 0:
                    scores_list.append(sc)
                    is_b_list.append(ib)
                    win_idx_list.append(np.full(n_p, w_idx, dtype=np.int32))
                    win_uniform_exp[w_idx] = float(np.sum(ib)) / n_p

                win_rank_only_is_b[w_idx] = 1.0 if rec["rank_only_is_b"] else 0.0

            if scores_list:
                cat_scores = np.concatenate(scores_list)
                cat_is_b = np.concatenate(is_b_list)
                cat_win = np.concatenate(win_idx_list)

                # Sort descending
                sort_idx = np.argsort(-cat_scores)
                sorted_scores = cat_scores[sort_idx]
                sorted_is_b = cat_is_b[sort_idx]
                sorted_win = cat_win[sort_idx]

                # Group boundaries where score changes
                diff = np.diff(sorted_scores)
                group_starts = np.concatenate([[0], np.where(diff != 0)[0] + 1])

                precomputed[c_name][s] = {
                    "has_data": True,
                    "sorted_is_b": sorted_is_b,
                    "sorted_win": sorted_win,
                    "group_starts": group_starts,
                    "win_n_pass": win_n_pass,
                    "win_rank_only_is_b": win_rank_only_is_b,
                    "win_uniform_exp": win_uniform_exp,
                }
            else:
                precomputed[c_name][s] = {"has_data": False}

    boot_ap_diff = {c_name: np.zeros(n_draws, dtype=np.float64) for c_name in cohort_names}
    boot_rank_diff = {c_name: np.zeros(n_draws, dtype=np.float64) for c_name in cohort_names}
    valid_draws = {c_name: 0 for c_name in cohort_names}

    for draw_idx in range(n_draws):
        sampled_seeds = rng.choice(seeds, size=3, replace=True)

        for c_name in cohort_names:
            meta = cohort_meta[c_name]
            n_tr = meta["n_tracks"]
            sampled_tr = rng.choice(n_tr, size=n_tr, replace=True)
            tr_multiplicity = np.bincount(sampled_tr, minlength=n_tr)

            win_multipliers = tr_multiplicity[meta["inv"]]
            win_w_sum = np.sum(win_multipliers)
            if win_w_sum <= 0:
                continue

            win_w_norm = win_multipliers / win_w_sum

            draw_ap_diffs = []
            draw_rank_diffs = []
            is_valid = True

            for s in sampled_seeds:
                p_data = precomputed[c_name][s]
                if not p_data["has_data"]:
                    is_valid = False
                    break

                win_n_p = p_data["win_n_pass"]
                # Action weights: win_w_norm / win_n_pass
                # Safe division where win_n_p > 0
                win_act_w = np.zeros_like(win_w_norm)
                nz_mask = (win_n_p > 0)
                win_act_w[nz_mask] = win_w_norm[nz_mask] / win_n_p[nz_mask]

                act_w = win_act_w[p_data["sorted_win"]]
                b_w = act_w * p_data["sorted_is_b"]

                gw = np.add.reduceat(act_w, p_data["group_starts"])
                gbw = np.add.reduceat(b_w, p_data["group_starts"])

                cum_w = np.cumsum(gw)
                cum_b_w = np.cumsum(gbw)

                tot_b = cum_b_w[-1]
                tot_w = cum_w[-1]

                if tot_b <= 0 or tot_w <= 0:
                    ap_val, base_val = 0.0, 0.0
                else:
                    nz_mask = (cum_w > 0)
                    prec = np.zeros_like(cum_w)
                    prec[nz_mask] = cum_b_w[nz_mask] / cum_w[nz_mask]
                    rec = gbw / tot_b
                    ap_val = float(np.sum(rec[nz_mask] * prec[nz_mask]))
                    base_val = float(tot_b / tot_w)

                if np.isnan(ap_val):
                    is_valid = False
                    break

                draw_ap_diffs.append(ap_val - base_val)

                rank_d = float(np.sum(win_w_norm * (p_data["win_rank_only_is_b"] - p_data["win_uniform_exp"])))
                draw_rank_diffs.append(rank_d)

            if is_valid and len(draw_ap_diffs) == 3:
                boot_ap_diff[c_name][valid_draws[c_name]] = float(np.mean(draw_ap_diffs))
                boot_rank_diff[c_name][valid_draws[c_name]] = float(np.mean(draw_rank_diffs))
                valid_draws[c_name] += 1

    alpha_bonf = 0.05 / 8.0
    q_low_bonf = alpha_bonf / 2.0
    q_high_bonf = 1.0 - (alpha_bonf / 2.0)

    summary = {}
    for c_name in cohort_names:
        n_v = valid_draws[c_name]
        if n_v < 9500:
            summary[c_name] = {
                "valid_draws": n_v,
                "status": "INSUFFICIENT_VALID_DRAWS",
            }
            continue

        arr_ap = boot_ap_diff[c_name][:n_v]
        arr_rank = boot_rank_diff[c_name][:n_v]

        summary[c_name] = {
            "valid_draws": n_v,
            "ap_diff": {
                "bonferroni_ci": [float(np.quantile(arr_ap, q_low_bonf)), float(np.quantile(arr_ap, q_high_bonf))],
                "ci_95": [float(np.quantile(arr_ap, 0.025)), float(np.quantile(arr_ap, 0.975))],
                "mean": float(np.mean(arr_ap)),
                "std": float(np.std(arr_ap)),
            },
            "rank_diff": {
                "bonferroni_ci": [float(np.quantile(arr_rank, q_low_bonf)), float(np.quantile(arr_rank, q_high_bonf))],
                "ci_95": [float(np.quantile(arr_rank, 0.025)), float(np.quantile(arr_rank, 0.975))],
                "mean": float(np.mean(arr_rank)),
                "std": float(np.std(arr_rank)),
            },
        }

    return summary


# Alias for compatibility
run_bootstrap_family8 = run_fast_bootstrap_family8
