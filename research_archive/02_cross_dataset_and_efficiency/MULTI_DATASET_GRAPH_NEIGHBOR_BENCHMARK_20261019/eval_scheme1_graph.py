"""Evaluation and Parameter Tuning for Scheme 1: Graph Interaction Optimization.
Runs all variations across the 100-sample multi-dataset benchmark.
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
)

def run_scheme1(device: torch.device):
    print("=" * 80)
    print("SCHEME 1: GRAPH INTERACTION MULTI-DATASET EVALUATION & TUNING")
    print("=" * 80)
    
    with (EXEC_DIR / "COHORT_100.json").open("r", encoding="utf-8") as f:
        cohort = json.load(f)
    
    model = prepare_model(device)
    builder = model.graph_builder
    orig_adaptive = builder._adaptive
    orig_build_context = builder.build_context

    def wrapped_build_context(trajectories, num_valid=None):
        comp = builder._history_components(trajectories)
        builder._last_components = comp
        return orig_build_context(trajectories, num_valid)

    builder.build_context = wrapped_build_context

    # Definitions of graph rules to compare
    def make_gated_adaptive(d_floor: float, v_close: float, th_conflict: float, th_tranquil: float):
        def _adaptive(score, candidates, valid):
            comp = getattr(builder, "_last_components", None)
            if comp is not None:
                dist = comp["distance"][:, -1]
                closing = comp["closing"][:, -1]
                cond = (dist <= d_floor) | (closing >= v_close)
                threshold = torch.where(cond, th_conflict, th_tranquil)
            else:
                threshold = th_tranquil

            num_nodes = score.shape[-1]
            adjacency = torch.zeros_like(candidates)
            keep = min(max(builder.k_max, 0), num_nodes)
            if keep == 0:
                return adjacency
            masked_score = score.masked_fill(~candidates, float("-inf"))
            ranked_values, ranked_indices = torch.topk(masked_score, keep, dim=-1)
            candidate_count = candidates.sum(dim=-1)

            above_threshold = (candidates & (score > threshold)).sum(dim=-1)
            desired = above_threshold.clamp(min=builder.k_min, max=builder.k_max)
            desired = torch.minimum(desired, candidate_count)
            rank = torch.arange(keep, device=score.device).view(1, 1, -1)
            selected = (rank < desired.unsqueeze(-1)) & torch.isfinite(ranked_values)
            adjacency.scatter_(dim=-1, index=ranked_indices, src=selected)
            return adjacency
        return _adaptive

    rules = [
        ("1.0_Default_Adaptive", "adaptive_kinematic", 0.35, 2, 4, 6.0, orig_adaptive),
        ("1.1_Thresh_0.25", "adaptive_kinematic", 0.25, 2, 4, 6.0, orig_adaptive),
        ("1.2_k_min_4", "adaptive_kinematic", 0.35, 4, 4, 6.0, orig_adaptive),
        ("1.3_Dist_Top4", "distance_topk", 0.35, 2, 4, 6.0, orig_adaptive),
        ("1.4_Full_Graph", "full", 0.35, 2, 4, 6.0, orig_adaptive),
        ("1.5_ConflictGated_v1", "adaptive_kinematic", 0.35, 2, 4, 6.0, make_gated_adaptive(2.0, 0.35, 0.25, 0.35)),
        ("1.6_ConflictGated_v2", "adaptive_kinematic", 0.35, 2, 4, 6.0, make_gated_adaptive(2.8, 0.30, 0.25, 0.35)),
        ("1.7_ConflictGated_v3_Radius8", "adaptive_kinematic", 0.35, 2, 4, 8.0, make_gated_adaptive(2.8, 0.25, 0.25, 0.35)),
    ]

    # Preload raw test arrays
    data_cache = {}
    for ds in cohort.keys():
        data_cache[ds] = {
            "data": np.load(RAW / f"{ds}_data_test.npy", mmap_mode="r"),
            "num": np.load(RAW / f"{ds}_num_test.npy"),
        }

    all_results = {r[0]: {ds: {"ade": [], "fde": [], "edges": []} for ds in cohort.keys()} for r in rules}

    t0 = time.perf_counter()
    for rule_name, mode, th, kmin, topk, radius, adaptive_fn in rules:
        builder.mode = mode
        builder.score_threshold = th
        builder.k_min = kmin
        builder.top_k = topk
        builder.radius = radius
        builder._adaptive = adaptive_fn

        for ds, samples in cohort.items():
            test_d = data_cache[ds]["data"]
            test_n = data_cache[ds]["num"]

            for item in samples:
                idx = item["sample_idx"]
                nv = int(test_n[idx])
                seq = np.array(test_d[idx], dtype=np.float32)
                past_w = seq[:9, :8, -2:].astype(np.float32)
                future_w = seq[:9, 8:, -2:].astype(np.float32)

                m, _, _, _, edge_info = forward_once(model, past_w, future_w, nv, device)
                all_results[rule_name][ds]["ade"].append(m["minADE20"])
                all_results[rule_name][ds]["fde"].append(m["minFDE20"])
                if "focal_recv_default" in edge_info:
                    edges = edge_info["focal_recv_default"]
                    all_results[rule_name][ds]["edges"].append(int(sum(edges[1:nv])))

    t_elapsed = time.perf_counter() - t0
    print(f"Evaluated {len(rules)} rules across 100 samples (5 datasets) in {t_elapsed:.2f}s!")

    # Summary table per dataset & overall
    summary = {}
    default_key = rules[0][0]
    
    print("\n" + "=" * 90)
    print(f"{'Graph Rule':<30} {'Overall minADE':<16} {'Overall minFDE':<16} {'Win/Tie/Loss vs Def':<20}")
    print("=" * 90)

    for rule_name, _, _, _, _, _, _ in rules:
        all_ade = []
        all_fde = []
        ds_stats = {}
        for ds in cohort.keys():
            ade_ds = all_results[rule_name][ds]["ade"]
            fde_ds = all_results[rule_name][ds]["fde"]
            all_ade.extend(ade_ds)
            all_fde.extend(fde_ds)
            ds_stats[ds] = {
                "mean_minADE": float(np.mean(ade_ds)),
                "mean_minFDE": float(np.mean(fde_ds)),
            }
        
        arr_ade = np.array(all_ade)
        arr_fde = np.array(all_fde)
        
        diff_ade = arr_ade - np.array([item for ds in cohort.keys() for item in all_results[default_key][ds]["ade"]])
        wins = int(np.sum(diff_ade < -1e-4))
        ties = int(np.sum(np.abs(diff_ade) <= 1e-4))
        losses = int(np.sum(diff_ade > 1e-4))

        summary[rule_name] = {
            "overall_minADE": float(np.mean(arr_ade)),
            "overall_minFDE": float(np.mean(arr_fde)),
            "per_dataset": ds_stats,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_delta_ade": float(np.mean(diff_ade)),
        }
        print(f"{rule_name:<30} {np.mean(arr_ade):<16.4f} {np.mean(arr_fde):<16.4f} {f'{wins}/{ties}/{losses}':<20}")
    print("=" * 90)

    out_file = EXEC_DIR / "SCHEME1_GRAPH_RESULTS.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Scheme 1 results to {out_file}")
    return summary

if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_scheme1(dev)
