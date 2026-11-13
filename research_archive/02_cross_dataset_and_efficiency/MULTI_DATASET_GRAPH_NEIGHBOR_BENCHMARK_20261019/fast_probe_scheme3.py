"""Fast Batched Probe comparing Scheme 1 and Scheme 3 across 5 datasets (500 samples each)."""
from __future__ import annotations

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

def eval_dataset_fast(model, ds_name: str, device: torch.device, l_min: float = 0.0, max_samples: int = 500, batch_size: int = 128):
    d = np.load(RAW / f"{ds_name}_data_test.npy", mmap_mode="r")
    num = np.load(RAW / f"{ds_name}_num_test.npy")
    total = min(len(num), max_samples)

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

            cands = pred[:, 0]
            gt = future_n[:, 0]
            err = torch.linalg.vector_norm(cands - gt.unsqueeze(1), dim=-1)
            ade = err.mean(dim=-1).cpu().numpy()
            fde = err[:, :, -1].cpu().numpy()

            all_min_ade.extend(ade.min(axis=-1).tolist())
            all_min_fde.extend(fde.min(axis=-1).tolist())

    return float(np.mean(all_min_ade)), float(np.mean(all_min_fde))

def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = prepare_model(dev)
    builder = model.graph_builder
    orig_adaptive = builder._adaptive
    orig_build_context = builder.build_context

    def wrapped_build_context(trajectories, num_valid=None):
        comp = builder._history_components(trajectories)
        builder._last_components = comp
        return orig_build_context(trajectories, num_valid)

    builder.build_context = wrapped_build_context

    configs = [
        ("Base (Adaptive, R=6.0)", 0.0, "adaptive_kinematic", 0.35, 2, 6.0, orig_adaptive),
        ("Scheme 1 (ConflictGated, kmin=4, R=6.0)", 0.0, "adaptive_kinematic", 0.35, 4, 6.0, make_conflict_gated_adaptive(builder)),
        ("Scheme 3.1 (Prune 0.05 + ConflictGated R=6.0)", 0.05, "adaptive_kinematic", 0.35, 4, 6.0, make_conflict_gated_adaptive(builder)),
        ("Scheme 3.2 (Prune 0.10 + ConflictGated R=6.0)", 0.10, "adaptive_kinematic", 0.35, 4, 6.0, make_conflict_gated_adaptive(builder)),
        ("Scheme 3.3 (Prune 0.10 + ConflictGated R=8.0)", 0.10, "adaptive_kinematic", 0.35, 4, 8.0, make_conflict_gated_adaptive(builder)),
    ]

    print("=" * 130)
    print(f"{'Configuration':<45} | {'ETH (364)':<14} | {'HOTEL (500)':<14} | {'UNIV (500)':<14} | {'ZARA01 (500)':<14} | {'ZARA02 (500)':<14} | {'Macro minADE':<10}")
    print("=" * 130)

    for name, l_min, mode, th, kmin, radius, adapt_fn in configs:
        builder.mode = mode
        builder.score_threshold = th
        builder.k_min = kmin
        builder.radius = radius
        builder._adaptive = adapt_fn

        res_ade = []
        res_fde = []
        line_str = f"{name:<45} | "
        for ds in DATASETS:
            ade, fde = eval_dataset_fast(model, ds, dev, l_min=l_min, max_samples=500, batch_size=128)
            res_ade.append(ade)
            res_fde.append(fde)
            line_str += f"{ade:.4f}/{fde:.4f} | "
        line_str += f"{np.mean(res_ade):.4f}"
        print(line_str)
    print("=" * 130)

if __name__ == "__main__":
    main()
