"""Complete ETH/UCY Benchmark Evaluation: Baseline vs Conflict-Gated (Scheme 1).
Evaluates all 5 test datasets: ETH, HOTEL, UNIV, ZARA01, ZARA02 (34,161 samples total).
Computes minADE20, minFDE20, top1_ADE, top1_FDE, pairwise wins/losses, and multi-agent subsets.
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
from CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018.NEIGHBOR_SWAP_6802.run_graph_force_6802 import prepare_model
from train_front_variant import normalize_to_focal_heading

DATASETS = ["eth", "hotel", "univ", "zara01", "zara02"]

def make_conflict_gated_adaptive(builder, d_floor=2.8, v_close=0.30, th_conflict=0.25, th_tranquil=0.35, k_min=4):
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
        desired = above_threshold.clamp(min=k_min, max=builder.k_max)
        desired = torch.minimum(desired, candidate_count)
        rank = torch.arange(keep, device=score.device).view(1, 1, -1)
        selected = (rank < desired.unsqueeze(-1)) & torch.isfinite(ranked_values)
        adjacency.scatter_(dim=-1, index=ranked_indices, src=selected)
        return adjacency
    return _adaptive

def eval_dataset(model, ds_name: str, device: torch.device, batch_size: int = 128):
    d = np.load(RAW / f"{ds_name}_data_test.npy", mmap_mode="r")
    num = np.load(RAW / f"{ds_name}_num_test.npy")
    total = len(num)

    all_min_ade = []
    all_min_fde = []
    all_top1_ade = []
    all_top1_fde = []
    all_nv = []

    with torch.inference_mode():
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            past = torch.from_numpy(d[i:end_idx, :9, :8, -2:].astype(np.float32)).to(device)
            future = torch.from_numpy(d[i:end_idx, :9, 8:, -2:].astype(np.float32)).to(device)
            nv = torch.from_numpy(num[i:end_idx].astype(np.int32)).to(device)

            past_n, future_n = normalize_to_focal_heading(past, future)
            center = past_n.mean(dim=-2, keepdim=True)
            pred = model(past_n - center, None, None, mode="testing", num_valid=nv, graph_loc=past_n)[2]
            pred = pred + center.unsqueeze(-3)

            cands = pred[:, 0]  # [B, 20, 12, 2]
            gt = future_n[:, 0]  # [B, 12, 2]
            err = torch.linalg.vector_norm(cands - gt.unsqueeze(1), dim=-1)  # [B, 20, 12]
            ade = err.mean(dim=-1).cpu().numpy()  # [B, 20]
            fde = err[:, :, -1].cpu().numpy()  # [B, 20]

            logits = model.last_mode_logits[:, 0].cpu().numpy()  # [B, 20]
            top1_idx = np.argmax(logits, axis=-1)
            b_range = np.arange(len(top1_idx))

            all_min_ade.extend(ade.min(axis=-1).tolist())
            all_min_fde.extend(fde.min(axis=-1).tolist())
            all_top1_ade.extend(ade[b_range, top1_idx].tolist())
            all_top1_fde.extend(fde[b_range, top1_idx].tolist())
            all_nv.extend(num[i:end_idx].tolist())

    return {
        "min_ade": np.array(all_min_ade, dtype=np.float64),
        "min_fde": np.array(all_min_fde, dtype=np.float64),
        "top1_ade": np.array(all_top1_ade, dtype=np.float64),
        "top1_fde": np.array(all_top1_fde, dtype=np.float64),
        "nv": np.array(all_nv, dtype=np.int32),
    }

def main():
    print("=" * 80)
    print("FULL ETH/UCY BENCHMARK EVALUATION (ALL 5 DATASETS)")
    print("Evaluating Baseline vs Conflict-Gated (Scheme 1)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = prepare_model(device)
    builder = model.graph_builder
    orig_adaptive = builder._adaptive
    orig_build_context = builder.build_context

    def wrapped_build_context(trajectories, num_valid=None):
        comp = builder._history_components(trajectories)
        builder._last_components = comp
        return orig_build_context(trajectories, num_valid)

    builder.build_context = wrapped_build_context

    # 1. Run Baseline across all 5 datasets
    print("\n--- Running Baseline (Default Adaptive: th=0.35, k_min=2, R=6.0m) ---")
    builder.mode = "adaptive_kinematic"
    builder.score_threshold = 0.35
    builder.k_min = 2
    builder.radius = 6.0
    builder._adaptive = orig_adaptive

    base_results = {}
    t0 = time.perf_counter()
    for ds in DATASETS:
        t_ds = time.perf_counter()
        res = eval_dataset(model, ds, device, batch_size=128)
        base_results[ds] = res
        print(f"[{ds:<8}] {len(res['nv']):<6} samples done in {time.perf_counter() - t_ds:.2f}s! minADE={np.mean(res['min_ade']):.4f}, minFDE={np.mean(res['min_fde']):.4f}")
    print(f"Baseline complete in {time.perf_counter() - t0:.2f}s!")

    # 2. Run Scheme 1: Conflict-Gated across all 5 datasets
    print("\n--- Running Scheme 1: Conflict-Gated Adaptive (d_floor=2.8m, v_close=0.30m/s, th_conflict=0.25, th_tranquil=0.35, k_min=4) ---")
    builder.mode = "adaptive_kinematic"
    builder.score_threshold = 0.35
    builder.k_min = 4
    builder.radius = 6.0
    builder._adaptive = make_conflict_gated_adaptive(builder, d_floor=2.8, v_close=0.30, th_conflict=0.25, th_tranquil=0.35, k_min=4)

    cg_results = {}
    t0_cg = time.perf_counter()
    for ds in DATASETS:
        t_ds = time.perf_counter()
        res = eval_dataset(model, ds, device, batch_size=128)
        cg_results[ds] = res
        print(f"[{ds:<8}] {len(res['nv']):<6} samples done in {time.perf_counter() - t_ds:.2f}s! minADE={np.mean(res['min_ade']):.4f}, minFDE={np.mean(res['min_fde']):.4f}")
    print(f"Conflict-Gated complete in {time.perf_counter() - t0_cg:.2f}s!")

    # 3. Compile Comparative Report
    report_data = {}
    print("\n" + "=" * 105)
    print(f"{'Dataset':<10} {'Samples':<8} {'Baseline minADE':<16} {'Conflict minADE':<16} {'ΔminADE':<10} {'Baseline minFDE':<16} {'Conflict minFDE':<16} {'ΔminFDE':<10} {'Win/Tie/Loss'}")
    print("=" * 105)

    all_base_ade, all_cg_ade = [], []
    all_base_fde, all_cg_fde = [], []
    all_base_t1_ade, all_cg_t1_ade = [], []
    all_base_t1_fde, all_cg_t1_fde = [], []

    multi_base_ade, multi_cg_ade = [], []
    multi_base_fde, multi_cg_fde = [], []

    for ds in DATASETS:
        b = base_results[ds]
        c = cg_results[ds]
        n_samples = len(b["nv"])

        b_ade_m, c_ade_m = float(np.mean(b["min_ade"])), float(np.mean(c["min_ade"]))
        b_fde_m, c_fde_m = float(np.mean(b["min_fde"])), float(np.mean(c["min_fde"]))
        b_t1_a, c_t1_a = float(np.mean(b["top1_ade"])), float(np.mean(c["top1_ade"]))
        b_t1_f, c_t1_f = float(np.mean(b["top1_fde"])), float(np.mean(c["top1_fde"]))

        d_ade = c_ade_m - b_ade_m
        d_fde = c_fde_m - b_fde_m

        diff = c["min_ade"] - b["min_ade"]
        wins = int(np.sum(diff < -1e-4))
        ties = int(np.sum(np.abs(diff) <= 1e-4))
        losses = int(np.sum(diff > 1e-4))
        w_t_l = f"{wins}/{ties}/{losses}"

        # Multi-agent subset (nv >= 2)
        m_mask = (b["nv"] >= 2)
        if np.any(m_mask):
            multi_base_ade.extend(b["min_ade"][m_mask].tolist())
            multi_cg_ade.extend(c["min_ade"][m_mask].tolist())
            multi_base_fde.extend(b["min_fde"][m_mask].tolist())
            multi_cg_fde.extend(c["min_fde"][m_mask].tolist())

        all_base_ade.extend(b["min_ade"].tolist())
        all_cg_ade.extend(c["min_ade"].tolist())
        all_base_fde.extend(b["min_fde"].tolist())
        all_cg_fde.extend(c["min_fde"].tolist())
        all_base_t1_ade.extend(b["top1_ade"].tolist())
        all_cg_t1_ade.extend(c["top1_ade"].tolist())
        all_base_t1_fde.extend(b["top1_fde"].tolist())
        all_cg_t1_fde.extend(c["top1_fde"].tolist())

        report_data[ds] = {
            "samples": n_samples,
            "baseline": {"min_ade": b_ade_m, "min_fde": b_fde_m, "top1_ade": b_t1_a, "top1_fde": b_t1_f},
            "conflict_gated": {"min_ade": c_ade_m, "min_fde": c_fde_m, "top1_ade": c_t1_a, "top1_fde": c_t1_f},
            "delta_min_ade": d_ade,
            "delta_min_fde": d_fde,
            "wins": wins, "ties": ties, "losses": losses,
        }

        print(f"{ds:<10} {n_samples:<8} {b_ade_m:<16.4f} {c_ade_m:<16.4f} {d_ade:<+10.4f} {b_fde_m:<16.4f} {c_fde_m:<16.4f} {d_fde:<+10.4f} {w_t_l}")

    print("=" * 105)

    # Macro Average (equal weight per dataset, standard benchmark protocol)
    macro_b_ade = float(np.mean([report_data[ds]["baseline"]["min_ade"] for ds in DATASETS]))
    macro_c_ade = float(np.mean([report_data[ds]["conflict_gated"]["min_ade"] for ds in DATASETS]))
    macro_b_fde = float(np.mean([report_data[ds]["baseline"]["min_fde"] for ds in DATASETS]))
    macro_c_fde = float(np.mean([report_data[ds]["conflict_gated"]["min_fde"] for ds in DATASETS]))

    # Micro Average (sample weighted across all 34,161 samples)
    micro_b_ade = float(np.mean(all_base_ade))
    micro_c_ade = float(np.mean(all_cg_ade))
    micro_b_fde = float(np.mean(all_base_fde))
    micro_c_fde = float(np.mean(all_cg_fde))

    total_diff = np.array(all_cg_ade) - np.array(all_base_ade)
    tot_wins = int(np.sum(total_diff < -1e-4))
    tot_ties = int(np.sum(np.abs(total_diff) <= 1e-4))
    tot_losses = int(np.sum(total_diff > 1e-4))

    print(f"{'MACRO AVG':<10} {'5 DS':<8} {macro_b_ade:<16.4f} {macro_c_ade:<16.4f} {macro_c_ade - macro_b_ade:<+10.4f} {macro_b_fde:<16.4f} {macro_c_fde:<16.4f} {macro_c_fde - macro_b_fde:<+10.4f} -")
    print(f"{'MICRO AVG':<10} {len(all_base_ade):<8} {micro_b_ade:<16.4f} {micro_c_ade:<16.4f} {micro_c_ade - micro_b_ade:<+10.4f} {micro_b_fde:<16.4f} {micro_c_fde:<16.4f} {micro_c_fde - micro_b_fde:<+10.4f} {tot_wins}/{tot_ties}/{tot_losses}")
    print("=" * 105)

    report_data["MACRO_AVG"] = {
        "baseline_min_ade": macro_b_ade, "conflict_min_ade": macro_c_ade, "delta_min_ade": macro_c_ade - macro_b_ade,
        "baseline_min_fde": macro_b_fde, "conflict_min_fde": macro_c_fde, "delta_min_fde": macro_c_fde - macro_b_fde,
    }
    report_data["MICRO_AVG"] = {
        "total_samples": len(all_base_ade),
        "baseline_min_ade": micro_b_ade, "conflict_min_ade": micro_c_ade, "delta_min_ade": micro_c_ade - micro_b_ade,
        "baseline_min_fde": micro_b_fde, "conflict_min_fde": micro_c_fde, "delta_min_fde": micro_c_fde - micro_b_fde,
        "total_wins": tot_wins, "total_ties": tot_ties, "total_losses": tot_losses,
    }

    out_file = EXEC_DIR / "ALL_ETH_UCY_BENCHMARK_RESULTS.json"
    out_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nSaved complete benchmark results to {out_file}")

if __name__ == "__main__":
    main()
