"""Master Training & Benchmark Script for Scheme 1 and Scheme 3 End-to-End Retraining.
1. Trains Scheme 1 (Conflict-Gated Adaptive) for 12 epochs on eth_data_train.npy (Seed 1, General).
2. Trains Scheme 3 (Co-adaptation: Static Prune + Conflict-Gated) for 12 epochs on eth_data_train.npy (Seed 1, General).
3. Evaluates both trained models across the full 34,161 ETH/UCY benchmark.
4. Audits Route B: Good Candidate Coverage (J <= 1.0), Rescued vs Lost, and Substitution Loss Rate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

EXEC_DIR = Path(__file__).resolve().parent
WORKSPACE = EXEC_DIR.parent
PROJECT = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
IR1 = WORKSPACE / "INTERACTION_RESIDUAL_SUPERVISION_SERIES"
SOURCE_ROOT = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "EXP_146B_ANCHORED_MODE_TEMPLATE" / "results"
RAW = PROJECT / "datasets/eth_ucy/processed_data_diverse"
PYTHON = sys.executable
TRAINER = FRONT / "train_front_variant.py"
DECODER_MODULE = IR1 / "current_baseline_adapter.py"
ENCODER_MODULE = EXEC_DIR / "scheme_encoder_adapter.py"

DATASETS = ["eth", "hotel", "univ", "zara01", "zara02"]

def run_training_experiment(scheme_name: str, exp_label: str):
    print("=" * 80)
    print(f"STARTING RETRAINING: {exp_label} ({scheme_name})")
    print("=" * 80)

    out_root = EXEC_DIR / "results" / "general" / scheme_name
    ckpt_path = out_root / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"
    if ckpt_path.is_file():
        print(f"[skip] {exp_label} already trained! Found checkpoint at: {ckpt_path}")
        return ckpt_path

    source_ckpt = SOURCE_ROOT / "general" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"
    if not source_ckpt.is_file():
        raise FileNotFoundError(f"Source checkpoint missing: {source_ckpt}")

    env = dict(os.environ)
    env["IRS_SOURCE_CHECKPOINT"] = str(source_ckpt)

    cmd = [
        PYTHON, "-u", str(TRAINER),
        "--project-root", str(PROJECT),
        "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(RAW / "eth_data_train.npy"),
        "--num-path", str(RAW / "eth_num_train.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "current_mode_anchor_template",
        "--decoder-module", str(DECODER_MODULE),
        "--encoder-experiment", scheme_name,
        "--encoder-module", str(ENCODER_MODULE),
        "--seed", "1",
        "--val-scene-id", "-1",
        "--validation-fraction", "0.05",
        "--validation-seed", "20260828",
        "--validation-split-mode", "stratified_random",
        "--epochs", "12",
        "--val-freq", "3",
        "--batch-size", "128",
        "--learning-rate", "0.00025",
        "--device", "cuda:0",
        "--max-agents", "9",
        "--graph-top-k", "4",
        "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine",
        "--warmup-epochs", "1",
        "--output-root", str(out_root),
        "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995",
        "--focal-heading-normalize",
        "--augment-reflection",
        "--augment-speed-min", "0.9",
        "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
        "--force",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT), capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {proc.returncode}")
    elapsed = time.perf_counter() - t0
    print(f"\nTraining for {exp_label} completed in {elapsed:.2f}s ({elapsed/60:.2f} min)!")

    ckpt_path = out_root / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    return ckpt_path

def evaluate_model_on_benchmark(ckpt_path: Path, scheme_name: str, l_min: float = 0.0):
    sys.path[:0] = [str(WORKSPACE), str(PROJECT), str(PROJECT / "scripts"), str(FRONT), str(IR1)]
    from CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018.NEIGHBOR_SWAP_6802.run_graph_force_6802 import prepare_model
    from train_front_variant import normalize_to_focal_heading
    from MULTI_DATASET_GRAPH_NEIGHBOR_BENCHMARK_20261019.audit_route_b_rescue_lost import (
        make_conflict_gated_adaptive, prune_static_batch
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = prepare_model(device)
    # Load newly trained weights into model
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    builder = model.graph_builder
    orig_adaptive = builder._adaptive
    orig_build_context = builder.build_context

    def wrapped_build_context(trajectories, num_valid=None):
        comp = builder._history_components(trajectories)
        builder._last_components = comp
        return orig_build_context(trajectories, num_valid)

    builder.build_context = wrapped_build_context
    builder.mode = "adaptive_kinematic"
    builder.score_threshold = 0.35
    builder.k_min = 4
    builder.radius = 6.0
    builder._adaptive = make_conflict_gated_adaptive(builder, d_floor=2.8, v_close=0.30, th_conflict=0.25, th_tranquil=0.35, k_min=4)

    all_j_dict = {}
    all_ade_dict = {}
    all_fde_dict = {}

    batch_size = 128
    with torch.inference_mode():
        for ds in DATASETS:
            d = np.load(RAW / f"{ds}_data_test.npy", mmap_mode="r")
            num = np.load(RAW / f"{ds}_num_test.npy")
            total = len(num)

            ds_j, ds_ade, ds_fde = [], [], []
            for i in range(0, total, batch_size):
                end_idx = min(i + batch_size, total)
                past = torch.from_numpy(d[i:end_idx, :9, :8, -2:].astype(np.float32)).to(device)
                future = torch.from_numpy(d[i:end_idx, :9, 8:, -2:].astype(np.float32)).to(device)
                nv = torch.from_numpy(num[i:end_idx].astype(np.int32)).to(device)

                if l_min > 0.0:
                    past, future, nv = prune_static_batch(past, future, nv, l_min)

                past_n, future_n = normalize_to_focal_heading(past, future)
                center = past_n.mean(dim=-2, keepdim=True)
                pred = model(past_n - center, None, None, mode="testing", num_valid=nv, graph_loc=past_n)[2]
                pred = pred + center.unsqueeze(-3)

                cands = pred[:, 0]
                gt = future_n[:, 0]
                err = torch.linalg.vector_norm(cands - gt.unsqueeze(1), dim=-1)
                ade = err.mean(dim=-1)
                fde = err[:, :, -1]
                j_modes = torch.maximum(ade / 0.20, fde / 0.40)
                j_sample = j_modes.min(dim=-1).values.cpu().numpy()

                ds_j.extend(j_sample.tolist())
                ds_ade.extend(ade.min(dim=-1).values.cpu().numpy().tolist())
                ds_fde.extend(fde.min(dim=-1).values.cpu().numpy().tolist())

            all_j_dict[ds] = np.array(ds_j, dtype=np.float64)
            all_ade_dict[ds] = np.array(ds_ade, dtype=np.float64)
            all_fde_dict[ds] = np.array(ds_fde, dtype=np.float64)
            print(f"[{scheme_name:<20}] {ds:<8}: minADE={np.mean(ds_ade):.4f}, minFDE={np.mean(ds_fde):.4f}, Cov={np.mean(np.array(ds_j) <= 1.0)*100:.2f}%")

    return all_j_dict, all_ade_dict, all_fde_dict

def analyze_against_base(base_j, new_j):
    b_good = base_j <= 1.0
    n_good = new_j <= 1.0
    rescued = int(np.sum((~b_good) & n_good))
    lost = int(np.sum(b_good & (~n_good)))
    net_rescue = rescued - lost
    churn_rate = (lost / max(rescued, 1)) * 100.0
    base_cov = float(np.mean(b_good) * 100.0)
    new_cov = float(np.mean(n_good) * 100.0)

    return {
        "base_coverage_pct": base_cov,
        "new_coverage_pct": new_cov,
        "delta_coverage_pct": new_cov - base_cov,
        "rescued": rescued,
        "lost": lost,
        "net_rescue": net_rescue,
        "churn_rate_pct": churn_rate,
        "mean_j_base": float(np.mean(base_j)),
        "mean_j_new": float(np.mean(new_j)),
    }

def main():
    print("=" * 80)
    print("STEP 1: TRAIN SCHEME 1 (CONFLICT-GATED ADAPTIVE GRAPH)")
    print("=" * 80)
    ckpt_s1 = run_training_experiment("scheme1_conflict_gated", "Scheme 1 (Conflict-Gated)")

    print("\n" + "=" * 80)
    print("STEP 2: TRAIN SCHEME 3 (CO-ADAPTIVE: STATIC PRUNE + CONFLICT-GATED GRAPH)")
    print("=" * 80)
    ckpt_s3 = run_training_experiment("scheme3_coadaptation", "Scheme 3 (Co-adaptation)")

    print("\n" + "=" * 80)
    print("STEP 3: BENCHMARK EVALUATION ACROSS ALL 34,161 SAMPLES")
    print("=" * 80)
    # Load base J metrics from Route B audit
    route_b_path = WORKSPACE / "MULTI_DATASET_GRAPH_NEIGHBOR_BENCHMARK_20261019" / "ROUTE_B_AUDIT_RESULTS.json"
    with route_b_path.open("r", encoding="utf-8") as f:
        route_b = json.load(f)

    # Evaluate retrained Scheme 1
    print("\nEvaluating Retrained Scheme 1...")
    s1_j, s1_ade, s1_fde = evaluate_model_on_benchmark(ckpt_s1, "Retrained_Scheme1", l_min=0.0)

    # Evaluate retrained Scheme 3
    print("\nEvaluating Retrained Scheme 3...")
    s3_j, s3_ade, s3_fde = evaluate_model_on_benchmark(ckpt_s3, "Retrained_Scheme3", l_min=0.10)

    # Compile Full Report
    summary = {"datasets": {}, "total": {}}

    # Base J
    sys.path[:0] = [str(WORKSPACE), str(PROJECT), str(PROJECT / "scripts"), str(FRONT), str(IR1)]
    from CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018.NEIGHBOR_SWAP_6802.run_graph_force_6802 import prepare_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_b = prepare_model(device)
    from MULTI_DATASET_GRAPH_NEIGHBOR_BENCHMARK_20261019.audit_route_b_rescue_lost import compute_j_metrics
    base_j_dict = {}
    base_ade_dict = {}
    base_fde_dict = {}
    for ds in DATASETS:
        j_arr, ade_arr, fde_arr = compute_j_metrics(model_b, ds, device, l_min=0.0)
        base_j_dict[ds] = j_arr
        base_ade_dict[ds] = ade_arr
        base_fde_dict[ds] = fde_arr

    all_b_j, all_s1_j, all_s3_j = [], [], []

    print("\n" + "=" * 125)
    print("RETRAINED SCHEME 1 vs BASELINE:")
    print(f"{'Dataset':<8} {'Samples':<8} {'Base Cov%':<11} {'New Cov%':<11} {'Delta Cov%':<12} {'Rescued':<10} {'Lost':<10} {'Net Rescue':<12} {'Churn Rate%':<14} {'minADE Delta'}")
    print("=" * 125)

    for ds in DATASETS:
        res1 = analyze_against_base(base_j_dict[ds], s1_j[ds])
        res3 = analyze_against_base(base_j_dict[ds], s3_j[ds])
        all_b_j.extend(base_j_dict[ds].tolist())
        all_s1_j.extend(s1_j[ds].tolist())
        all_s3_j.extend(s3_j[ds].tolist())

        d_ade = np.mean(s1_ade[ds]) - np.mean(base_ade_dict[ds])
        print(f"{ds:<8} {len(base_j_dict[ds]):<8} {res1['base_coverage_pct']:<11.2f} {res1['new_coverage_pct']:<11.2f} {res1['delta_coverage_pct']:<+12.2f} {res1['rescued']:<10} {res1['lost']:<10} {res1['net_rescue']:<+12} {res1['churn_rate_pct']:<14.1f} {d_ade*1000:<+.2f} mm")

        summary["datasets"][ds] = {
            "samples": len(base_j_dict[ds]),
            "scheme1": res1,
            "scheme3": res3,
            "s1_min_ade": float(np.mean(s1_ade[ds])),
            "s1_min_fde": float(np.mean(s1_fde[ds])),
            "s3_min_ade": float(np.mean(s3_ade[ds])),
            "s3_min_fde": float(np.mean(s3_fde[ds])),
            "base_min_ade": float(np.mean(base_ade_dict[ds])),
            "base_min_fde": float(np.mean(base_fde_dict[ds])),
        }

    tot_s1 = analyze_against_base(np.array(all_b_j), np.array(all_s1_j))
    tot_s3 = analyze_against_base(np.array(all_b_j), np.array(all_s3_j))
    summary["total"]["scheme1"] = tot_s1
    summary["total"]["scheme3"] = tot_s3

    print("-" * 125)
    print(f"{'TOTAL':<8} {len(all_b_j):<8} {tot_s1['base_coverage_pct']:<11.2f} {tot_s1['new_coverage_pct']:<11.2f} {tot_s1['delta_coverage_pct']:<+12.2f} {tot_s1['rescued']:<10} {tot_s1['lost']:<10} {tot_s1['net_rescue']:<+12} {tot_s1['churn_rate_pct']:<14.1f} —")
    print("=" * 125)

    print("\n" + "=" * 125)
    print("RETRAINED SCHEME 3 vs BASELINE:")
    print(f"{'Dataset':<8} {'Samples':<8} {'Base Cov%':<11} {'New Cov%':<11} {'Delta Cov%':<12} {'Rescued':<10} {'Lost':<10} {'Net Rescue':<12} {'Churn Rate%':<14} {'minADE Delta'}")
    print("=" * 125)

    for ds in DATASETS:
        res3 = summary["datasets"][ds]["scheme3"]
        d_ade = np.mean(s3_ade[ds]) - np.mean(base_ade_dict[ds])
        print(f"{ds:<8} {len(base_j_dict[ds]):<8} {res3['base_coverage_pct']:<11.2f} {res3['new_coverage_pct']:<11.2f} {res3['delta_coverage_pct']:<+12.2f} {res3['rescued']:<10} {res3['lost']:<10} {res3['net_rescue']:<+12} {res3['churn_rate_pct']:<14.1f} {d_ade*1000:<+.2f} mm")

    print("-" * 125)
    print(f"{'TOTAL':<8} {len(all_b_j):<8} {tot_s3['base_coverage_pct']:<11.2f} {tot_s3['new_coverage_pct']:<11.2f} {tot_s3['delta_coverage_pct']:<+12.2f} {tot_s3['rescued']:<10} {tot_s3['lost']:<10} {tot_s3['net_rescue']:<+12} {tot_s3['churn_rate_pct']:<14.1f} —")
    print("=" * 125)

    out_file = EXEC_DIR / "RETRAINED_SCHEMES_BENCHMARK_RESULTS.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nRetraining & Evaluation Benchmark saved to {out_file}")

if __name__ == "__main__":
    main()
