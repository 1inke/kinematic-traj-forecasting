"""Encoder adapter for Scheme 1 (Conflict-Gated) and Scheme 3 (Co-adaptation) retraining."""
from __future__ import annotations

import types
import torch

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

def prune_static_inputs(past_loc, fut_loc, num_valid, graph_loc, l_min=0.10):
    if l_min <= 0.0 or num_valid is None:
        return past_loc, fut_loc, num_valid, graph_loc

    B = past_loc.shape[0]
    device = past_loc.device

    new_past = torch.zeros_like(past_loc)
    new_fut = torch.zeros_like(fut_loc) if fut_loc is not None else None
    new_graph = torch.zeros_like(graph_loc) if graph_loc is not None else None
    new_nv = torch.zeros_like(num_valid)

    # Use graph_loc if available to judge real-world movement before center normalization
    ref_past = graph_loc if graph_loc is not None else past_loc
    diff = ref_past[:, 1:, 1:] - ref_past[:, 1:, :-1]
    lens = torch.norm(diff, dim=-1).sum(dim=-1)  # [B, 8]

    for b in range(B):
        nv = num_valid[b].item()
        new_past[b, 0] = past_loc[b, 0]
        if new_fut is not None:
            new_fut[b, 0] = fut_loc[b, 0]
        if new_graph is not None:
            new_graph[b, 0] = graph_loc[b, 0]

        if nv > 1:
            cur_lens = lens[b, :nv - 1]
            moving = torch.where(cur_lens >= l_min)[0]
            k = len(moving)
            if k > 0:
                new_past[b, 1:1 + k] = past_loc[b, 1 + moving]
                if new_fut is not None:
                    new_fut[b, 1:1 + k] = fut_loc[b, 1 + moving]
                if new_graph is not None:
                    new_graph[b, 1:1 + k] = graph_loc[b, 1 + moving]
            new_nv[b] = 1 + k
        else:
            new_nv[b] = 1

    return new_past, new_fut, new_nv, new_graph

def replace_encoder(model, cfg, name: str) -> None:
    builder = model.graph_builder
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

    if name == "scheme1_conflict_gated":
        print(f"[replace_encoder] Configured Scheme 1: Conflict-Gated Adaptive (k_min=4, th_conflict=0.25, R=6.0m)")
    elif name == "scheme3_coadaptation":
        orig_forward = model.forward.__func__

        def scheme3_forward(self, past_loc, fut_loc=None, unet_features=None, mode="train", num_valid=None, graph_loc=None):
            p, f, nv, g = prune_static_inputs(past_loc, fut_loc, num_valid, graph_loc, l_min=0.10)
            return orig_forward(self, p, f, unet_features, mode=mode, num_valid=nv, graph_loc=g)

        model.forward = types.MethodType(scheme3_forward, model)
        print(f"[replace_encoder] Configured Scheme 3: Co-adaptation (Static Prune 0.10m + Conflict-Gated k_min=4)")
    else:
        raise ValueError(f"Unknown encoder experiment: {name}")
