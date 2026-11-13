"""Evaluation and Parameter Tuning for Scheme 3: Co-adaptation (Coupled Neighbor + Graph).
Combines kinematic neighbor selection with velocity-compensated adaptive graph rules.
Evaluates across Case 6802 and the 100-sample multi-dataset benchmark.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch

EXEC_DIR = Path(__file__).resolve().parent
WORKSPACE = EXEC_DIR.parent
PROJECT = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
IR1 = WORKSPACE / "INTERACTION_RESIDUAL_SUPERVISION_SERIES"
RAW = PROJECT / "datasets/eth_ucy/processed_data_diverse"

import sys
sys.path[:0] = [str(WORKSPACE), str(PROJECT), str(PROJECT / "scripts"), str(FRONT), str(IR1)]
import sitecustomize
from experiment_config import load_locked_q20
from CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018.NEIGHBOR_SWAP_6802.run_graph_force_6802 import (
    prepare_model,
    forward_once,
    load_label_xy,
    build_past_filled,
    build_future,
    OBS_FRAMES,
    PRED_FRAMES,
    SAMPLE_ID,
)

def run_scheme3(device: torch.device):
    print("=" * 80)
    print("SCHEME 3: CO-ADAPTIVE NEIGHBOR + GRAPH MULTI-DATASET EVALUATION")
    print("=" * 80)

    model = prepare_model(device)
    builder = model.graph_builder
    orig_adaptive = builder._adaptive
    orig_build_context = builder.build_context

    def wrapped_build_context(trajectories, num_valid=None):
        comp = builder._history_components(trajectories)
        builder._last_components = comp
        return orig_build_context(trajectories, num_valid)

    builder.build_context = wrapped_build_context

    # 1. In-depth Co-adaptation Tuning on Case 6802
    table = load_label_xy([130, 148, 131, 146, 147, 144, 145, 70, 111, 143, 151, 150])
    focal_p, _, _ = build_past_filled(table, 130, OBS_FRAMES)
    focal_f = build_future(table, 130, PRED_FRAMES, focal_p[-1])

    all_peds = [148, 131, 146, 147, 144, 145, 70, 111, 143, 151, 150]
    ped_past = {p: build_past_filled(table, p, OBS_FRAMES)[0] for p in all_peds}
    ped_fut = {p: build_future(table, p, PRED_FRAMES, ped_past[p][-1]) for p in all_peds}

    def make_tensor(p_list):
        p_tensor = np.zeros((9, 8, 2), dtype=np.float32)
        f_tensor = np.zeros((9, 12, 2), dtype=np.float32)
        p_tensor[0] = focal_p
        f_tensor[0] = focal_f
        for i, p in enumerate(p_list[:8]):
            p_tensor[i + 1] = ped_past[p]
            f_tensor[i + 1] = ped_fut[p]
        return p_tensor, f_tensor, min(9, 1 + len(p_list))

    # Test combinations on 6802:
    # Neighbor sets:
    # A) Base: [148, 131, 146, 147, 144, 145, 70, 111]
    # B) Smart Dynamic: [148, 131, 146, 147, 144, 145, 143, 151]
    # C) High Conflict Priority: [148, 131, 146, 147, 143, 151, 150, 144]
    p_base = [148, 131, 146, 147, 144, 145, 70, 111]
    p_dyn = [148, 131, 146, 147, 144, 145, 143, 151]
    p_conf = [148, 131, 146, 147, 143, 151, 150, 144]

    # Graph rules for co-adaptation:
    def make_dynamic_radius_adaptive(gamma: float = 2.5, th: float = 0.25, kmin: int = 4):
        def _adaptive(score, candidates, valid):
            comp = getattr(builder, "_last_components", None)
            if comp is not None:
                dist = comp["distance"][:, -1]
                closing = comp["closing"][:, -1]
                cond = (dist <= 2.8) | (closing >= 0.30)
                threshold = torch.where(cond, th, 0.35)
            else:
                threshold = 0.35

            num_nodes = score.shape[-1]
            adjacency = torch.zeros_like(candidates)
            keep = min(max(builder.k_max, 0), num_nodes)
            if keep == 0: return adjacency
            masked_score = score.masked_fill(~candidates, float("-inf"))
            ranked_values, ranked_indices = torch.topk(masked_score, keep, dim=-1)
            candidate_count = candidates.sum(dim=-1)

            above_threshold = (candidates & (score > threshold)).sum(dim=-1)
            desired = above_threshold.clamp(min=kmin, max=builder.k_max)
            desired = torch.minimum(desired, candidate_count)
            rank = torch.arange(keep, device=score.device).view(1, 1, -1)
            selected = (rank < desired.unsqueeze(-1)) & torch.isfinite(ranked_values)
            adjacency.scatter_(dim=-1, index=ranked_indices, src=selected)
            return adjacency
        return _adaptive

    co_configs_6802 = [
        ("Base Neigh + Default Graph", p_base, "adaptive_kinematic", 0.35, 2, 6.0, orig_adaptive),
        ("Smart Neigh + Default Graph", p_dyn, "adaptive_kinematic", 0.35, 2, 6.0, orig_adaptive),
        ("Base Neigh + Gated Graph (kmin=4)", p_base, "adaptive_kinematic", 0.25, 4, 6.0, make_dynamic_radius_adaptive(0.0, 0.25, 4)),
        ("Smart Neigh + Gated Graph (R=8.5, kmin=4)", p_dyn, "adaptive_kinematic", 0.25, 4, 8.5, make_dynamic_radius_adaptive(2.5, 0.25, 4)),
        ("Conflict Neigh + Gated Graph (R=8.5, kmin=4)", p_conf, "adaptive_kinematic", 0.25, 4, 8.5, make_dynamic_radius_adaptive(2.5, 0.25, 4)),
        ("Conflict Neigh + Dist Top-6 (R=9.0)", p_conf, "distance_topk", 0.35, 2, 9.0, orig_adaptive),
        ("Smart Neigh + Full Graph", p_dyn, "full", 0.35, 2, 9.0, orig_adaptive),
        ("Conflict Neigh + Full Graph", p_conf, "full", 0.35, 2, 9.0, orig_adaptive),
    ]

    print("\nCase 6802 Co-adaptation Matrix:")
    print(f"{'Configuration':<45} {'minADE':<9} {'minFDE':<9} {'top1_ADE':<10} {'top1_FDE':<10}")
    print("-" * 85)
    results_6802 = {}
    for name, p_list, mode, th, kmin, radius, adapt_fn in co_configs_6802:
        builder.mode = mode
        builder.score_threshold = th
        builder.k_min = kmin
        builder.radius = radius
        builder._adaptive = adapt_fn
        if mode == "distance_topk":
            builder.top_k = 6

        p_t, f_t, nv = make_tensor(p_list)
        m, _, _, _, _ = forward_once(model, p_t, f_t, nv, device)
        results_6802[name] = m
        print(f"{name:<45} {m['minADE20']:<9.4f} {m['minFDE20']:<9.4f} {m['top1_ADE']:<10.4f} {m['top1_FDE']:<10.4f}")
    print("-" * 85)

    # 2. Multi-Dataset 100-Sample Benchmark Evaluation for Scheme 3
    with (EXEC_DIR / "COHORT_100.json").open("r", encoding="utf-8") as f:
        cohort = json.load(f)

    data_cache = {}
    for ds in cohort.keys():
        data_cache[ds] = {
            "data": np.load(RAW / f"{ds}_data_test.npy", mmap_mode="r"),
            "num": np.load(RAW / f"{ds}_num_test.npy"),
        }

    # Parallel Co-adaptive Pipelines on Cohort:
    # Pipeline 1: Default Baseline
    # Pipeline 2: Static Filtered + Conflict Gated Adaptive (kmin=4, R=6.0)
    # Pipeline 3: Static Filtered + Velocity Compensated Radius (R=8.0, kmin=4)
    # Pipeline 4: Static Filtered + Distance Top-4
    pipeline_configs = [
        ("3.0_Default_Baseline", False, "adaptive_kinematic", 0.35, 2, 6.0, orig_adaptive),
        ("3.1_CoAdapt_Gated_kmin4", True, "adaptive_kinematic", 0.25, 4, 6.0, make_dynamic_radius_adaptive(0.0, 0.25, 4)),
        ("3.2_CoAdapt_VeloComp_R8", True, "adaptive_kinematic", 0.25, 4, 8.0, make_dynamic_radius_adaptive(2.5, 0.25, 4)),
        ("3.3_CoAdapt_DistTop4", True, "distance_topk", 0.35, 2, 6.0, orig_adaptive),
    ]

    pipeline_results = {p[0]: {ds: {"ade": [], "fde": []} for ds in cohort.keys()} for p in pipeline_configs}

    t0 = time.perf_counter()
    for p_name, do_filter, mode, th, kmin, radius, adapt_fn in pipeline_configs:
        builder.mode = mode
        builder.score_threshold = th
        builder.k_min = kmin
        builder.radius = radius
        builder._adaptive = adapt_fn
        if mode == "distance_topk":
            builder.top_k = 4

        for ds, samples in cohort.items():
            test_d = data_cache[ds]["data"]
            test_n = data_cache[ds]["num"]

            for item in samples:
                idx = item["sample_idx"]
                nv = int(test_n[idx])
                seq = np.array(test_d[idx], dtype=np.float32)
                past_w = seq[:9, :8, -2:].astype(np.float32)
                future_w = seq[:9, 8:, -2:].astype(np.float32)

                if do_filter and nv > 1:
                    path_lens = np.linalg.norm(np.diff(past_w[1:nv], axis=1), axis=-1).sum(axis=-1)
                    perm = np.argsort(-path_lens)
                    past_w[1:nv] = past_w[1:nv][perm]
                    future_w[1:nv] = future_w[1:nv][perm]

                m, _, _, _, _ = forward_once(model, past_w, future_w, nv, device)
                pipeline_results[p_name][ds]["ade"].append(m["minADE20"])
                pipeline_results[p_name][ds]["fde"].append(m["minFDE20"])

    t_elapsed = time.perf_counter() - t0
    print(f"\nCohort evaluation complete in {t_elapsed:.2f}s!")

    summary_s3 = {"case_6802": results_6802, "cohort_summary": {}}
    default_p = pipeline_configs[0][0]

    print("\n" + "=" * 90)
    print(f"{'Co-adaptive Pipeline':<30} {'Overall minADE':<16} {'Overall minFDE':<16} {'Win/Tie/Loss vs Base':<20}")
    print("=" * 90)

    for p_name, _, _, _, _, _, _ in pipeline_configs:
        all_ade = []
        all_fde = []
        ds_stats = {}
        for ds in cohort.keys():
            ades = pipeline_results[p_name][ds]["ade"]
            fdes = pipeline_results[p_name][ds]["fde"]
            all_ade.extend(ades)
            all_fde.extend(fdes)
            ds_stats[ds] = {
                "mean_minADE": float(np.mean(ades)),
                "mean_minFDE": float(np.mean(fdes)),
            }

        arr_ade = np.array(all_ade)
        arr_fde = np.array(all_fde)
        diff_ade = arr_ade - np.array([item for ds in cohort.keys() for item in pipeline_results[default_p][ds]["ade"]])
        wins = int(np.sum(diff_ade < -1e-4))
        ties = int(np.sum(np.abs(diff_ade) <= 1e-4))
        losses = int(np.sum(diff_ade > 1e-4))

        summary_s3["cohort_summary"][p_name] = {
            "overall_minADE": float(np.mean(arr_ade)),
            "overall_minFDE": float(np.mean(arr_fde)),
            "per_dataset": ds_stats,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_delta_ade": float(np.mean(diff_ade)),
        }
        print(f"{p_name:<30} {np.mean(arr_ade):<16.4f} {np.mean(arr_fde):<16.4f} {f'{wins}/{ties}/{losses}':<20}")
    print("=" * 90)

    out_file = EXEC_DIR / "SCHEME3_COADAPTATION_RESULTS.json"
    out_file.write_text(json.dumps(summary_s3, indent=2), encoding="utf-8")
    print(f"Saved Scheme 3 results to {out_file}")
    return summary_s3

if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_scheme3(dev)
