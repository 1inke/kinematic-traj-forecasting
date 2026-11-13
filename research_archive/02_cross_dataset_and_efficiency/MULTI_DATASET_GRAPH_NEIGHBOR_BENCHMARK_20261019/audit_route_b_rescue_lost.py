"""Route B Quantitative Audit: Complete Good Candidate Coverage and Net Rescue (Rescue vs Lost).
Evaluates Baseline, Scheme 1 (Conflict-Gated), and Scheme 3 (Co-adaptation) across all 34,161 test samples.
Criterion from GEMINI.md:
  Good Candidate: ADE <= 0.20m AND FDE <= 0.40m
  J = min_k max(ADE_k / 0.20, FDE_k / 0.40)
  Success: J <= 1.0
  Rescued: J_base > 1.0 and J_scheme <= 1.0
  Lost: J_base <= 1.0 and J_scheme > 1.0
  Net Rescue = Rescued - Lost
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

def prune_static_batch(past: torch.Tensor, future: torch.Tensor, nv: torch.Tensor, l_min: float):
    if l_min <= 0.0:
        return past, future, nv
    B = past.shape[0]
    new_past = torch.zeros_like(past)
    new_future = torch.zeros_like(future)
    new_nv = torch.zeros_like(nv)

    diff = past[:, 1:, 1:] - past[:, 1:, :-1]
    lens = torch.norm(diff, dim=-1).sum(dim=-1)

    for b in range(B):
        cur_nv = nv[b].item()
        new_past[b, 0] = past[b, 0]
        new_future[b, 0] = future[b, 0]
        if cur_nv > 1:
            cur_lens = lens[b, :cur_nv - 1]
            moving = torch.where(cur_lens >= l_min)[0]
            k = len(moving)
            if k > 0:
                new_past[b, 1:1 + k] = past[b, 1 + moving]
                new_future[b, 1:1 + k] = future[b, 1 + moving]
            new_nv[b] = 1 + k
        else:
            new_nv[b] = 1
    return new_past, new_future, new_nv

def compute_j_metrics(model, ds_name: str, device: torch.device, l_min: float = 0.0, batch_size: int = 128):
    d = np.load(RAW / f"{ds_name}_data_test.npy", mmap_mode="r")
    num = np.load(RAW / f"{ds_name}_num_test.npy")
    total = len(num)

    all_j = []
    all_min_ade = []
    all_min_fde = []

    with torch.inference_mode():
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

            cands = pred[:, 0]  # [B, 20, 12, 2]
            gt = future_n[:, 0]  # [B, 12, 2]
            err = torch.linalg.vector_norm(cands - gt.unsqueeze(1), dim=-1)  # [B, 20, 12]
            ade_modes = err.mean(dim=-1)  # [B, 20]
            fde_modes = err[:, :, -1]  # [B, 20]

            # Joint metric J_k = max(ADE_k / 0.20, FDE_k / 0.40)
            j_modes = torch.maximum(ade_modes / 0.20, fde_modes / 0.40)  # [B, 20]
            j_sample = j_modes.min(dim=-1).values.cpu().numpy()  # [B]

            all_j.extend(j_sample.tolist())
            all_min_ade.extend(ade_modes.min(dim=-1).values.cpu().numpy().tolist())
            all_min_fde.extend(fde_modes.min(dim=-1).values.cpu().numpy().tolist())

    return np.array(all_j, dtype=np.float64), np.array(all_min_ade, dtype=np.float64), np.array(all_min_fde, dtype=np.float64)

def analyze_scheme(j_base, j_scheme):
    good_base = j_base <= 1.0
    good_scheme = j_scheme <= 1.0

    rescued = int(np.sum((~good_base) & good_scheme))
    lost = int(np.sum(good_base & (~good_scheme)))
    net_rescue = rescued - lost
    base_cov = float(np.mean(good_base) * 100.0)
    scheme_cov = float(np.mean(good_scheme) * 100.0)
    mean_j_base = float(np.mean(j_base))
    mean_j_scheme = float(np.mean(j_scheme))

    return {
        "base_coverage_pct": base_cov,
        "scheme_coverage_pct": scheme_cov,
        "delta_coverage_pct": scheme_cov - base_cov,
        "rescued": rescued,
        "lost": lost,
        "net_rescue": net_rescue,
        "mean_j_base": mean_j_base,
        "mean_j_scheme": mean_j_scheme,
        "delta_mean_j": mean_j_scheme - mean_j_base,
    }

def main():
    print("=" * 90)
    print("ROUTE B AUDIT: COMPLETE GOOD CANDIDATE COVERAGE & RESCUE VS LOST ANALYSIS")
    print("Evaluating all 34,161 samples across 5 datasets (ETH, HOTEL, UNIV, ZARA01, ZARA02)")
    print("=" * 90)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = prepare_model(device)
    builder = model.graph_builder
    orig_adaptive = builder._adaptive
    orig_build_context = builder.build_context

    def wrapped_build_context(trajectories, num_valid=None):
        comp = builder._history_components(trajectories)
        builder._last_components = comp
        return orig_build_context(trajectories, num_valid)

    builder.build_context = wrapped_build_context

    # 1. Evaluate Baseline
    print("\n1. Running Baseline (th=0.35, k_min=2, R=6.0m)...")
    builder.mode = "adaptive_kinematic"
    builder.score_threshold = 0.35
    builder.k_min = 2
    builder.radius = 6.0
    builder._adaptive = orig_adaptive

    base_j = {}
    for ds in DATASETS:
        j_arr, _, _ = compute_j_metrics(model, ds, device, l_min=0.0)
        base_j[ds] = j_arr
        print(f"[{ds:<8}] Coverage: {np.mean(j_arr <= 1.0)*100:.2f}%, Mean J: {np.mean(j_arr):.4f}")

    # 2. Evaluate Scheme 1: Conflict-Gated
    print("\n2. Running Scheme 1 (Conflict-Gated Adaptive, k_min=4)...")
    builder.mode = "adaptive_kinematic"
    builder.score_threshold = 0.35
    builder.k_min = 4
    builder.radius = 6.0
    builder._adaptive = make_conflict_gated_adaptive(builder, d_floor=2.8, v_close=0.30, th_conflict=0.25, th_tranquil=0.35, k_min=4)

    s1_j = {}
    for ds in DATASETS:
        j_arr, _, _ = compute_j_metrics(model, ds, device, l_min=0.0)
        s1_j[ds] = j_arr
        print(f"[{ds:<8}] Coverage: {np.mean(j_arr <= 1.0)*100:.2f}%, Mean J: {np.mean(j_arr):.4f}")

    # 3. Evaluate Scheme 3: Co-adaptation (Static Pruning 0.10m + Conflict-Gated)
    print("\n3. Running Scheme 3 (Static Prune 0.10m + Conflict-Gated, k_min=4)...")
    s3_j = {}
    for ds in DATASETS:
        j_arr, _, _ = compute_j_metrics(model, ds, device, l_min=0.10)
        s3_j[ds] = j_arr
        print(f"[{ds:<8}] Coverage: {np.mean(j_arr <= 1.0)*100:.2f}%, Mean J: {np.mean(j_arr):.4f}")

    # 4. Cross-Dataset Analysis and Reporting
    audit_results = {"datasets": {}, "total_summary": {}}

    print("\n" + "=" * 115)
    print("SCHEME 1 (CONFLICT-GATED) vs BASELINE AUDIT:")
    print(f"{'Dataset':<8} {'Samples':<8} {'Base Cov%':<11} {'S1 Cov%':<11} {'Delta Cov%':<12} {'Rescued':<10} {'Lost':<10} {'Net Rescue':<12} {'Mean J Delta'}")
    print("=" * 115)

    all_b_j, all_s1_j, all_s3_j = [], [], []

    for ds in DATASETS:
        res1 = analyze_scheme(base_j[ds], s1_j[ds])
        res3 = analyze_scheme(base_j[ds], s3_j[ds])
        audit_results["datasets"][ds] = {"scheme1": res1, "scheme3": res3, "samples": len(base_j[ds])}

        all_b_j.extend(base_j[ds].tolist())
        all_s1_j.extend(s1_j[ds].tolist())
        all_s3_j.extend(s3_j[ds].tolist())

        n = len(base_j[ds])
        print(f"{ds:<8} {n:<8} {res1['base_coverage_pct']:<11.2f} {res1['scheme_coverage_pct']:<11.2f} {res1['delta_coverage_pct']:<+12.2f} {res1['rescued']:<10} {res1['lost']:<10} {res1['net_rescue']:<+12} {res1['delta_mean_j']:<+.4f}")

    tot_s1 = analyze_scheme(np.array(all_b_j), np.array(all_s1_j))
    tot_s3 = analyze_scheme(np.array(all_b_j), np.array(all_s3_j))
    audit_results["total_summary"]["scheme1"] = tot_s1
    audit_results["total_summary"]["scheme3"] = tot_s3

    print("-" * 115)
    print(f"{'TOTAL':<8} {len(all_b_j):<8} {tot_s1['base_coverage_pct']:<11.2f} {tot_s1['scheme_coverage_pct']:<11.2f} {tot_s1['delta_coverage_pct']:<+12.2f} {tot_s1['rescued']:<10} {tot_s1['lost']:<10} {tot_s1['net_rescue']:<+12} {tot_s1['delta_mean_j']:<+.4f}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("SCHEME 3 (CO-ADAPTATION) vs BASELINE AUDIT:")
    print(f"{'Dataset':<8} {'Samples':<8} {'Base Cov%':<11} {'S3 Cov%':<11} {'Delta Cov%':<12} {'Rescued':<10} {'Lost':<10} {'Net Rescue':<12} {'Mean J Delta'}")
    print("=" * 115)

    for ds in DATASETS:
        res3 = audit_results["datasets"][ds]["scheme3"]
        n = len(base_j[ds])
        print(f"{ds:<8} {n:<8} {res3['base_coverage_pct']:<11.2f} {res3['scheme_coverage_pct']:<11.2f} {res3['delta_coverage_pct']:<+12.2f} {res3['rescued']:<10} {res3['lost']:<10} {res3['net_rescue']:<+12} {res3['delta_mean_j']:<+.4f}")

    print("-" * 115)
    print(f"{'TOTAL':<8} {len(all_b_j):<8} {tot_s3['base_coverage_pct']:<11.2f} {tot_s3['scheme_coverage_pct']:<11.2f} {tot_s3['delta_coverage_pct']:<+12.2f} {tot_s3['rescued']:<10} {tot_s3['lost']:<10} {tot_s3['net_rescue']:<+12} {tot_s3['delta_mean_j']:<+.4f}")
    print("=" * 115)

    out_file = EXEC_DIR / "ROUTE_B_AUDIT_RESULTS.json"
    out_file.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    print(f"\nRoute B Audit saved to {out_file}")

if __name__ == "__main__":
    main()
