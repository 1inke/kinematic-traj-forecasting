"""Full ETH/UCY Benchmark Evaluation for Scheme 3: Co-adaptation.
Evaluates across all 5 test datasets (34,161 samples): ETH, HOTEL, UNIV, ZARA01, ZARA02.
Compares Scheme 3 (Static Pruning + Conflict-Gated Adaptive) against Baseline and Scheme 1.
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
    lens = torch.norm(diff, dim=-1).sum(dim=-1)  # [B, 8]

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

def eval_dataset(model, ds_name: str, device: torch.device, l_min: float = 0.0, batch_size: int = 128):
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

            if l_min > 0.0:
                past, future, nv = prune_static_batch(past, future, nv, l_min)

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
    print("FULL ETH/UCY BENCHMARK EVALUATION (ALL 5 DATASETS: 34,161 SAMPLES)")
    print("Evaluating Scheme 3: Co-adaptation (Static Pruning + Conflict-Gated Adaptive)")
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

    # 1. Run Baseline
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
        res = eval_dataset(model, ds, device, l_min=0.0, batch_size=128)
        base_results[ds] = res
        print(f"[{ds:<8}] {len(res['nv']):<6} samples done in {time.perf_counter() - t_ds:.2f}s! minADE={np.mean(res['min_ade']):.4f}, minFDE={np.mean(res['min_fde']):.4f}")
    print(f"Baseline complete in {time.perf_counter() - t0:.2f}s!")

    # 2. Run Scheme 3: Co-adaptation (Static Prune L_min=0.10m + Conflict-Gated Adaptive)
    print("\n--- Running Scheme 3: Co-adaptation (Static Prune 0.10m + Conflict-Gated kmin=4) ---")
    builder.mode = "adaptive_kinematic"
    builder.score_threshold = 0.35
    builder.k_min = 4
    builder.radius = 6.0
    builder._adaptive = make_conflict_gated_adaptive(builder, d_floor=2.8, v_close=0.30, th_conflict=0.25, th_tranquil=0.35, k_min=4)

    s3_results = {}
    t0_s3 = time.perf_counter()
    for ds in DATASETS:
        t_ds = time.perf_counter()
        res = eval_dataset(model, ds, device, l_min=0.10, batch_size=128)
        s3_results[ds] = res
        print(f"[{ds:<8}] {len(res['nv']):<6} samples done in {time.perf_counter() - t_ds:.2f}s! minADE={np.mean(res['min_ade']):.4f}, minFDE={np.mean(res['min_fde']):.4f}")
    print(f"Scheme 3 complete in {time.perf_counter() - t0_s3:.2f}s!")

    # 3. Compile Comparative Report
    report = {}
    macro_base_ade, macro_base_fde = [], []
    macro_s3_ade, macro_s3_fde = [], []

    total_samples = 0
    micro_base_ade, micro_base_fde = [], []
    micro_s3_ade, micro_s3_fde = [], []
    total_wins, total_ties, total_losses = 0, 0, 0

    print("\n" + "=" * 110)
    print(f"{'Dataset':<8} {'Samples':<8} {'Base minADE':<12} {'S3 minADE':<12} {'Delta ADE (mm)':<15} {'Base minFDE':<12} {'S3 minFDE':<12} {'Delta FDE (mm)':<15} {'Wins/Ties/Losses'}")
    print("=" * 110)

    for ds in DATASETS:
        b_ade = base_results[ds]["min_ade"]
        b_fde = base_results[ds]["min_fde"]
        s_ade = s3_results[ds]["min_ade"]
        s_fde = s3_results[ds]["min_fde"]

        n = len(b_ade)
        total_samples += n

        m_b_ade, m_s_ade = float(np.mean(b_ade)), float(np.mean(s_ade))
        m_b_fde, m_s_fde = float(np.mean(b_fde)), float(np.mean(s_fde))

        macro_base_ade.append(m_b_ade)
        macro_base_fde.append(m_b_fde)
        macro_s3_ade.append(m_s_ade)
        macro_s3_fde.append(m_s_fde)

        micro_base_ade.extend(b_ade.tolist())
        micro_base_fde.extend(b_fde.tolist())
        micro_s3_ade.extend(s_ade.tolist())
        micro_s3_fde.extend(s_fde.tolist())

        diff = s_ade - b_ade
        wins = int(np.sum(diff < -1e-4))
        ties = int(np.sum(np.abs(diff) <= 1e-4))
        losses = int(np.sum(diff > 1e-4))

        total_wins += wins
        total_ties += ties
        total_losses += losses

        delta_ade_mm = (m_s_ade - m_b_ade) * 1000.0
        delta_fde_mm = (m_s_fde - m_b_fde) * 1000.0

        print(f"{ds:<8} {n:<8} {m_b_ade:<12.4f} {m_s_ade:<12.4f} {delta_ade_mm:<+15.2f} {m_b_fde:<12.4f} {m_s_fde:<12.4f} {delta_fde_mm:<+15.2f} {f'{wins}/{ties}/{losses}'}")

        report[ds] = {
            "samples": n,
            "baseline": {
                "min_ade": m_b_ade,
                "min_fde": m_b_fde,
                "top1_ade": float(np.mean(base_results[ds]["top1_ade"])),
                "top1_fde": float(np.mean(base_results[ds]["top1_fde"])),
            },
            "scheme3": {
                "min_ade": m_s_ade,
                "min_fde": m_s_fde,
                "top1_ade": float(np.mean(s3_results[ds]["top1_ade"])),
                "top1_fde": float(np.mean(s3_results[ds]["top1_fde"])),
            },
            "delta_min_ade": m_s_ade - m_b_ade,
            "delta_min_fde": m_s_fde - m_b_fde,
            "delta_min_ade_mm": delta_ade_mm,
            "delta_min_fde_mm": delta_fde_mm,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }

    macro_b_ade = float(np.mean(macro_base_ade))
    macro_s_ade = float(np.mean(macro_s3_ade))
    macro_b_fde = float(np.mean(macro_base_fde))
    macro_s_fde = float(np.mean(macro_s3_fde))

    micro_b_ade = float(np.mean(micro_base_ade))
    micro_s_ade = float(np.mean(micro_s3_ade))
    micro_b_fde = float(np.mean(micro_base_fde))
    micro_s_fde = float(np.mean(micro_s3_fde))

    print("-" * 110)
    print(f"{'MACRO':<8} {5:<8} {macro_b_ade:<12.4f} {macro_s_ade:<12.4f} {(macro_s_ade - macro_b_ade)*1000:<+15.2f} {macro_b_fde:<12.4f} {macro_s_fde:<12.4f} {(macro_s_fde - macro_b_fde)*1000:<+15.2f} —")
    print(f"{'MICRO':<8} {total_samples:<8} {micro_b_ade:<12.4f} {micro_s_ade:<12.4f} {(micro_s_ade - micro_b_ade)*1000:<+15.2f} {micro_b_fde:<12.4f} {micro_s_fde:<12.4f} {(micro_s_fde - micro_b_fde)*1000:<+15.2f} {f'{total_wins}/{total_ties}/{total_losses}'}")
    print("=" * 110)

    report["MACRO_AVG"] = {
        "baseline_min_ade": macro_b_ade,
        "scheme3_min_ade": macro_s_ade,
        "delta_min_ade": macro_s_ade - macro_b_ade,
        "delta_min_ade_mm": (macro_s_ade - macro_b_ade) * 1000.0,
        "baseline_min_fde": macro_b_fde,
        "scheme3_min_fde": macro_s_fde,
        "delta_min_fde": macro_s_fde - macro_b_fde,
        "delta_min_fde_mm": (macro_s_fde - macro_b_fde) * 1000.0,
    }

    report["MICRO_AVG"] = {
        "total_samples": total_samples,
        "baseline_min_ade": micro_b_ade,
        "scheme3_min_ade": micro_s_ade,
        "delta_min_ade": micro_s_ade - micro_b_ade,
        "delta_min_ade_mm": (micro_s_ade - micro_b_ade) * 1000.0,
        "baseline_min_fde": micro_b_fde,
        "scheme3_min_fde": micro_s_fde,
        "delta_min_fde": micro_s_fde - micro_b_fde,
        "delta_min_fde_mm": (micro_s_fde - micro_b_fde) * 1000.0,
        "total_wins": total_wins,
        "total_ties": total_ties,
        "total_losses": total_losses,
    }

    out_file = EXEC_DIR / "ALL_ETH_UCY_SCHEME3_RESULTS.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAll results saved to {out_file}")

if __name__ == "__main__":
    main()
