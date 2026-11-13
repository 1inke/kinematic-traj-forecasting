"""Probe script for Scheme 3 co-adaptation parameter tuning across datasets.
Tests static suppression thresholds (L_min) and radius variations coupled with conflict-gated graph.
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

def eval_subset(model, ds_name: str, device: torch.device, l_min: float = 0.0, num_samples: int = 400):
    d = np.load(RAW / f"{ds_name}_data_test.npy", mmap_mode="r")
    num = np.load(RAW / f"{ds_name}_num_test.npy")
    total = min(len(num), num_samples)

    all_min_ade = []
    all_min_fde = []

    with torch.inference_mode():
        for i in range(total):
            nv = int(num[i])
            past_np = d[i, :9, :8, -2:].astype(np.float32)
            fut_np = d[i, :9, 8:, -2:].astype(np.float32)

            if l_min > 0.0 and nv > 1:
                path_lens = np.linalg.norm(np.diff(past_np[1:nv], axis=1), axis=-1).sum(axis=-1)
                moving_idx = np.where(path_lens >= l_min)[0]
                p_new = np.zeros_like(past_np)
                f_new = np.zeros_like(fut_np)
                p_new[0] = past_np[0]
                f_new[0] = fut_np[0]
                for n_idx, o_idx in enumerate(moving_idx):
                    p_new[n_idx + 1] = past_np[o_idx + 1]
                    f_new[n_idx + 1] = fut_np[o_idx + 1]
                past_np = p_new
                fut_np = f_new
                nv = 1 + len(moving_idx)

            past = torch.from_numpy(past_np[None]).to(device)
            future = torch.from_numpy(fut_np[None]).to(device)
            nv_t = torch.tensor([nv], dtype=torch.int32, device=device)

            past_n, future_n = normalize_to_focal_heading(past, future)
            center = past_n.mean(dim=-2, keepdim=True)
            pred = model(past_n - center, None, None, mode="testing", num_valid=nv_t, graph_loc=past_n)[2]
            pred = pred + center.unsqueeze(-3)

            cands = pred[:, 0]
            gt = future_n[:, 0]
            err = torch.linalg.vector_norm(cands - gt.unsqueeze(1), dim=-1)
            ade = err.mean(dim=-1).cpu().numpy()
            fde = err[:, :, -1].cpu().numpy()

            all_min_ade.append(ade.min())
            all_min_fde.append(fde.min())

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
        ("Base (Default Adaptive, R=6.0)", 0.0, "adaptive_kinematic", 0.35, 2, 6.0, orig_adaptive),
        ("Scheme1 (Conflict-Gated, kmin=4, R=6.0)", 0.0, "adaptive_kinematic", 0.35, 4, 6.0, make_conflict_gated_adaptive(builder)),
        ("Scheme3A (Static Prune 0.05 + Conflict-Gated)", 0.05, "adaptive_kinematic", 0.35, 4, 6.0, make_conflict_gated_adaptive(builder)),
        ("Scheme3B (Static Prune 0.10 + Conflict-Gated)", 0.10, "adaptive_kinematic", 0.35, 4, 6.0, make_conflict_gated_adaptive(builder)),
        ("Scheme3C (Static Prune 0.10 + Conflict-Gated R=8.0)", 0.10, "adaptive_kinematic", 0.35, 4, 8.0, make_conflict_gated_adaptive(builder)),
    ]

    print(f"{'Config':<45} | {'ETH':<13} | {'HOTEL':<13} | {'UNIV':<13} | {'ZARA01':<13} | {'ZARA02':<13} | {'Avg minADE':<10}")
    print("-" * 135)

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
            ade, fde = eval_subset(model, ds, dev, l_min=l_min, num_samples=300)
            res_ade.append(ade)
            res_fde.append(fde)
            line_str += f"{ade:.4f}/{fde:.4f} | "
        line_str += f"{np.mean(res_ade):.4f}"
        print(line_str)

if __name__ == "__main__":
    main()
