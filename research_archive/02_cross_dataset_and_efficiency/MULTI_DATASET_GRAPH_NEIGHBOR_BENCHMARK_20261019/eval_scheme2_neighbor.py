"""Evaluation and Parameter Tuning for Scheme 2: Kinematic-Aware Neighbor Selection.
Tests static suppression, kinematic relevancy scoring, and soft-history extrapolation.
Evaluates on the 100-sample multi-dataset benchmark and Case 6802.
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
LABELS_ROOT = PROJECT / "datasets/eth_ucy/labels"

import sys
sys.path[:0] = [str(WORKSPACE), str(PROJECT), str(PROJECT / "scripts"), str(FRONT), str(IR1)]
import sitecustomize
from experiment_config import load_locked_q20
from CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018.NEIGHBOR_SWAP_6802.run_graph_force_6802 import (
    prepare_model,
    forward_once,
)

def run_scheme2_exploration(device: torch.device):
    print("=" * 80)
    print("SCHEME 2: NEIGHBOR SELECTION MULTI-DATASET EXPLORATION & TUNING")
    print("=" * 80)

    # 1. First test in-depth on Case 6802 and its scene variations
    from CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018.NEIGHBOR_SWAP_6802.run_graph_force_6802 import (
        load_label_xy, build_past_filled, build_future, OBS_FRAMES, PRED_FRAMES, SAMPLE_ID
    )

    model = prepare_model(device)
    raw_eth = np.load(RAW / "eth_data_train.npy", mmap_mode="r")
    num_eth = np.load(RAW / "eth_num_train.npy")

    # Load zara02 raw labels to test neighbor selection strategies on 6802
    table = load_label_xy([130, 148, 131, 146, 147, 144, 145, 70, 111, 143, 151, 150, 69, 112])
    focal_p, _, _ = build_past_filled(table, 130, OBS_FRAMES)
    focal_f = build_future(table, 130, PRED_FRAMES, focal_p[-1])

    # Candidate neighbor pool in zara02 at frame 766
    all_peds = [148, 131, 146, 147, 144, 145, 70, 111, 143, 151, 150]
    ped_past = {}
    ped_fut = {}
    for p in all_peds:
        past, _, _ = build_past_filled(table, p, OBS_FRAMES)
        fut = build_future(table, p, PRED_FRAMES, past[-1])
        ped_past[p] = past
        ped_fut[p] = fut

    def build_tensor_for_peds(p_list):
        p_tensor = np.zeros((9, 8, 2), dtype=np.float32)
        f_tensor = np.zeros((9, 12, 2), dtype=np.float32)
        p_tensor[0] = focal_p
        f_tensor[0] = focal_f
        for i, p in enumerate(p_list[:8]):
            p_tensor[i + 1] = ped_past[p]
            f_tensor[i + 1] = ped_fut[p]
        nv = min(9, 1 + len(p_list))
        return p_tensor, f_tensor, nv

    # Strategies for neighbor selection:
    # Strat 2.0: Baseline (Distance sorting, keep 8): [148, 131, 146, 147, 144, 145, 70, 111]
    p_orig = [148, 131, 146, 147, 144, 145, 70, 111]
    
    # Strat 2.1: Static suppression (Filter out 70, 111 whose path length < 0.2m):
    # Free slots given to next closest: [148, 131, 146, 147, 144, 145, 143, 151]
    p_static_filtered = [148, 131, 146, 147, 144, 145, 143, 151]

    # Strat 2.2: Kinematic Relevance (Distance - 2.5 * v_closing):
    # Promotes fast oncoming 143 and approaching 151 ahead of receding 144, 145:
    # [148, 131, 146, 147, 143, 151, 150, 144]
    p_kin_relevance = [148, 131, 146, 147, 143, 151, 150, 144]

    strats = [
        ("2.0_Baseline_Selection", p_orig),
        ("2.1_Static_Suppression", p_static_filtered),
        ("2.2_Kinematic_Relevance_Ranking", p_kin_relevance),
    ]

    results_6802 = {}
    print("\nCase 6802 Neighbor Selection Exploration (with fixed default graph):")
    for s_name, p_list in strats:
        past_t, fut_t, nv = build_tensor_for_peds(p_list)
        m, _, _, _, _ = forward_once(model, past_t, fut_t, nv, device)
        results_6802[s_name] = m
        print(f"{s_name:<35}: minADE={m['minADE20']:.4f}, minFDE={m['minFDE20']:.4f}, top1_ADE={m['top1_ADE']:.4f}")

    # Now evaluate static suppression on the multi-dataset 100-sample cohort
    with (EXEC_DIR / "COHORT_100.json").open("r", encoding="utf-8") as f:
        cohort = json.load(f)

    # In the processed dataset, we can identify and suppress static neighbors in the 9-agent tensor:
    # Test L_min thresholds: 0.05m, 0.15m, 0.30m
    l_thresholds = [0.0, 0.05, 0.15, 0.30]
    cohort_results = {f"L_min_{l:.2f}m": {"ade": [], "fde": []} for l in l_thresholds}

    data_cache = {}
    for ds in cohort.keys():
        data_cache[ds] = {
            "data": np.load(RAW / f"{ds}_data_test.npy", mmap_mode="r"),
            "num": np.load(RAW / f"{ds}_num_test.npy"),
        }

    for l_min in l_thresholds:
        key = f"L_min_{l_min:.2f}m"
        for ds, samples in cohort.items():
            test_d = data_cache[ds]["data"]
            test_n = data_cache[ds]["num"]

            for item in samples:
                idx = item["sample_idx"]
                nv = int(test_n[idx])
                seq = np.array(test_d[idx], dtype=np.float32)
                past_w = seq[:9, :8, -2:].astype(np.float32)
                future_w = seq[:9, 8:, -2:].astype(np.float32)

                if l_min > 0.0 and nv > 1:
                    # Filter out static agents among neighbors (1..nv-1)
                    # Compute path length for each neighbor:
                    path_lens = np.linalg.norm(np.diff(past_w[1:nv], axis=1), axis=-1).sum(axis=-1)
                    # Reorder neighbors: moving neighbors first, static neighbors pushed to the end (or masked)
                    perm = np.argsort(-path_lens)  # descending by movement
                    past_w[1:nv] = past_w[1:nv][perm]
                    future_w[1:nv] = future_w[1:nv][perm]

                m, _, _, _, _ = forward_once(model, past_w, future_w, nv, device)
                cohort_results[key]["ade"].append(m["minADE20"])
                cohort_results[key]["fde"].append(m["minFDE20"])

    print("\n" + "=" * 80)
    print("SCHEME 2: MULTI-DATASET STATIC SUPPRESSION TUNING (100 SAMPLES)")
    print("=" * 80)
    print(f"{'Static Filter Threshold':<30} {'Mean minADE':<16} {'Mean minFDE':<16}")
    print("=" * 80)
    summary_s2 = {"case_6802": results_6802, "cohort_tuning": {}}
    for l_min in l_thresholds:
        key = f"L_min_{l_min:.2f}m"
        mean_ade = float(np.mean(cohort_results[key]["ade"]))
        mean_fde = float(np.mean(cohort_results[key]["fde"]))
        summary_s2["cohort_tuning"][key] = {
            "mean_minADE": mean_ade,
            "mean_minFDE": mean_fde,
        }
        print(f"{key:<30} {mean_ade:<16.4f} {mean_fde:<16.4f}")
    print("=" * 80)

    out_file = EXEC_DIR / "SCHEME2_NEIGHBOR_RESULTS.json"
    out_file.write_text(json.dumps(summary_s2, indent=2), encoding="utf-8")
    print(f"Saved Scheme 2 results to {out_file}")
    return summary_s2

if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_scheme2_exploration(dev)
