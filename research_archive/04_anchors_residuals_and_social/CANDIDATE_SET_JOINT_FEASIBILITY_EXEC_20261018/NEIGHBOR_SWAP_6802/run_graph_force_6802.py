# -*- coding: utf-8 -*-
"""Force graph edges for 6802: focal <-> close neighbors + BR 151/150.

Extends neighbor-swap: keep F16 IR1 weights, put 151/150 in agent tensor
(padded past), then OR-force focal receive edges past radius/Top-K.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

EXEC = Path(r"<WORKSPACE_ROOT>/CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
OUT = EXEC / "NEIGHBOR_SWAP_6802"
OUT.mkdir(exist_ok=True)

WORKSPACE = Path(r"<WORKSPACE_ROOT>")
PROJECT = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
IR1 = WORKSPACE / "INTERACTION_RESIDUAL_SUPERVISION_SERIES"
SOURCE_ROOT = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "EXP_146B_ANCHORED_MODE_TEMPLATE" / "results"
RAW = PROJECT / "datasets/eth_ucy/processed_data_diverse"
LABEL = PROJECT / "datasets/eth_ucy/labels/zara02/crowds_zara02.txt"

SAMPLE_ID = 6802
SEED = 1
SPLIT = "general"
STORED_PEDS = [130, 148, 131, 146, 147, 144, 145, 70, 111]
OBS_FRAMES = list(range(759, 767))
PRED_FRAMES = list(range(767, 779))
LAMBDA_FDE = 0.5
# After swap: agents 7,8 hold 150,151; 1..4 are close 148/131/146/147
FORCE_FOCAL_TO = None  # set in main after knowing slots


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_label_xy(ped_ids):
    want = set(int(p) for p in ped_ids)
    table = {}
    with LABEL.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 16:
                continue
            frame = int(float(parts[0]))
            ped = int(float(parts[1]))
            if ped not in want:
                continue
            table.setdefault(frame, {})[ped] = np.array(
                [float(parts[13]), float(parts[15])], dtype=np.float64
            )
    return table


def path_length(traj: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())


def build_past_filled(table, ped, frames):
    known = [(f, table[f][ped]) for f in frames if f in table and ped in table[f]]
    if not known:
        raise ValueError(f"ped {ped} missing")
    first_xy = known[0][1]
    out = np.zeros((len(frames), 2), dtype=np.float32)
    filled = 0
    for i, f in enumerate(frames):
        if f in table and ped in table[f]:
            out[i] = table[f][ped]
        else:
            out[i] = first_xy
            filled += 1
    return out, filled, known[0][0]


def build_future(table, ped, frames, fallback):
    out = np.zeros((len(frames), 2), dtype=np.float32)
    for i, f in enumerate(frames):
        if f in table and ped in table[f]:
            out[i] = table[f][ped]
        else:
            out[i] = out[i - 1] if i else fallback
    return out


def metrics_from_preds(cands, gt, logits):
    err = np.linalg.norm(cands - gt[None], axis=-1)
    ade, fde = err.mean(-1), err[:, -1]
    cost = ade + LAMBDA_FDE * fde
    top1 = int(np.argmax(logits))
    return {
        "minADE20": float(ade.min()),
        "minFDE20": float(fde.min()),
        "oracle_cost": float(cost.min()),
        "best_ade_mode": int(ade.argmin()),
        "top1_mode": top1,
        "top1_ADE": float(ade[top1]),
        "top1_FDE": float(fde[top1]),
        "ade_all": ade.tolist(),
        "fde_all": fde.tolist(),
    }


def force_focal_neighbors_allowed(allowed: torch.Tensor, neighbor_ids: list[int]) -> torch.Tensor:
    """Keep non-focal rows; set focal receive row to self + forced neighbor cols."""
    masked = allowed.clone()
    if masked.ndim == 3:
        # [B,A,A]
        masked[:, 0, :] = False
        masked[:, 0, 0] = True
        for j in neighbor_ids:
            masked[:, 0, j] = True
    elif masked.ndim == 4:
        # [B,H,A,A]
        masked[:, :, 0, :] = False
        masked[:, :, 0, 0] = True
        for j in neighbor_ids:
            masked[:, :, 0, j] = True
    else:
        raise ValueError(tuple(masked.shape))
    return masked


@contextmanager
def force_focal_edges(model, neighbor_ids: list[int]):
    encoder = model.agent_aware_trans_encoder
    if not bool(getattr(encoder, "use_temporal_social_cross", False)):
        raise RuntimeError("need temporal_social_cross")
    original = encoder.temporal_social_attention

    def wrapped(self, temporal, agent, relation, allowed, type_features=None, risk_score=None):
        self.last_force_original_allowed = allowed.detach().clone()
        forced = force_focal_neighbors_allowed(allowed, neighbor_ids)
        self.last_force_masked_allowed = forced.detach().clone()
        return original(temporal, agent, relation, forced, type_features, risk_score)

    encoder.temporal_social_attention = types.MethodType(wrapped, encoder)
    try:
        yield model
    finally:
        encoder.temporal_social_attention = original


def prepare_model(device):
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT), str(IR1)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA

    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.MODEL.DECODER.NUM_MODES = 20
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]

    source = (
        SOURCE_ROOT / SPLIT / "f16_temporal_social_cross" / f"seed_{SEED}" / "val_scene_-1" / "best_selected.pth"
    )
    ckpt = (
        IR1 / "results" / SPLIT / "baseline" / "f16_temporal_social_cross" / f"seed_{SEED}" / "val_scene_-1" / "best_selected.pth"
    )
    os.environ["IRS_SOURCE_CHECKPOINT"] = str(source)
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    adapter = load_module("csna_adapter_force", IR1 / "current_baseline_adapter.py")
    adapter.replace_decoder(model, cfg, "current_mode_anchor_template")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def forward_once(model, past_xy, future_xy, num_valid, device, force_ids=None):
    from train_front_variant import normalize_to_focal_heading

    past = torch.from_numpy(past_xy[None].astype(np.float32)).to(device)
    future = torch.from_numpy(future_xy[None].astype(np.float32)).to(device)
    nv = torch.tensor([num_valid], dtype=torch.int32, device=device)
    past_n, future_n = normalize_to_focal_heading(past, future)
    center = past_n.mean(dim=-2, keepdim=True)

    ctx = force_focal_edges(model, force_ids) if force_ids is not None else nullctx(model)
    with ctx:
        pred = model(past_n - center, None, None, mode="testing", num_valid=nv, graph_loc=past_n)[2]
    pred = pred + center.unsqueeze(-3)
    cands_h = pred[0, 0].cpu().numpy()
    gt_h = future_n[0, 0].cpu().numpy()
    logits = model.last_mode_logits[0, 0].cpu().numpy()

    # world invert
    focal = past_xy[0]
    origin = focal[-1].astype(np.float64)
    disp = focal[-1] - focal[-4]
    speed = float(np.linalg.norm(disp))
    heading = (disp / speed) if speed > 0.03 else np.array([1.0, 0.0])
    c, s = float(heading[0]), float(heading[1])

    def inv_rot(p):
        xl, yl = p[..., 0], p[..., 1]
        return np.stack([xl * c - yl * s, xl * s + yl * c], axis=-1)

    cands_w = inv_rot(cands_h) + origin
    gt_w = inv_rot(gt_h) + origin
    m = metrics_from_preds(cands_h, gt_h, logits)
    m["best_end_world"] = cands_w[m["best_ade_mode"], -1].tolist()
    m["top1_end_world"] = cands_w[m["top1_mode"], -1].tolist()
    m["end_gt_world"] = gt_w[-1].tolist()

    # edge snapshot
    enc = model.agent_aware_trans_encoder
    edge_info = {}
    if force_ids is not None and hasattr(enc, "last_force_original_allowed"):
        orig = enc.last_force_original_allowed[0]
        forced = enc.last_force_masked_allowed[0]
        if orig.ndim == 3:  # unexpected
            pass
        # [A,A] or [H,A,A]
        if orig.ndim == 3:
            o_row = orig[:, 0, :].any(0) if False else orig[0]
            # actually [H,A,A] -> take any head
            o_focal = orig.any(0)[0].cpu().numpy()
            f_focal = forced.any(0)[0].cpu().numpy()
        else:
            o_focal = orig[0].cpu().numpy()
            f_focal = forced[0].cpu().numpy()
        edge_info = {
            "focal_recv_before": o_focal.astype(bool).tolist(),
            "focal_recv_after": f_focal.astype(bool).tolist(),
            "forced_ids": force_ids,
        }
    elif hasattr(enc, "last_allowed") and enc.last_allowed is not None:
        al = enc.last_allowed[0]
        if al.ndim == 3:
            row = al.any(0)[0].cpu().numpy()
        else:
            row = al[0].cpu().numpy()
        edge_info = {"focal_recv_default": row.astype(bool).tolist()}

    return m, cands_w, gt_w, logits, edge_info


class nullctx:
    def __init__(self, v):
        self.v = v

    def __enter__(self):
        return self.v

    def __exit__(self, *a):
        return False


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    raw = np.load(RAW / "eth_data_train.npy", mmap_mode="r")
    num = np.load(RAW / "eth_num_train.npy")
    nv = int(num[SAMPLE_ID])
    seq = np.array(raw[SAMPLE_ID])
    past_w = seq[:nv, :8, -2:].astype(np.float32)
    future_w = seq[:nv, 8:, -2:].astype(np.float32)

    table = load_label_xy(set(STORED_PEDS + [151, 150]))
    # match peds
    matched = []
    for a in range(nv):
        xy = past_w[a, -1]
        best, bestd = None, 1e9
        for ped in STORED_PEDS:
            if 766 in table and ped in table[766]:
                d = float(np.linalg.norm(xy - table[766][ped]))
                if d < bestd:
                    bestd, best = d, ped
        matched.append({"agent": a, "ped": best, "path_len": path_length(past_w[a])})
    assert matched[0]["ped"] == 130

    neigh_stats = sorted(matched[1:], key=lambda r: r["path_len"])
    drop_agents = [neigh_stats[0]["agent"], neigh_stats[1]["agent"]]
    drop_peds = [neigh_stats[0]["ped"], neigh_stats[1]["ped"]]
    print("drop", drop_agents, drop_peds)

    p151, fill151, first151 = build_past_filled(table, 151, OBS_FRAMES)
    p150, fill150, first150 = build_past_filled(table, 150, OBS_FRAMES)
    f151 = build_future(table, 151, PRED_FRAMES, p151[-1])
    f150 = build_future(table, 150, PRED_FRAMES, p150[-1])

    past_p = past_w.copy()
    future_p = future_w.copy()
    # put 151 in first drop slot, 150 in second
    past_p[drop_agents[0]] = p151
    past_p[drop_agents[1]] = p150
    future_p[drop_agents[0]] = f151
    future_p[drop_agents[1]] = f150
    new_peds = STORED_PEDS.copy()
    new_peds[drop_agents[0]] = 151
    new_peds[drop_agents[1]] = 150

    # agent indices for close + BR
    ped_to_agent_base = {m["ped"]: m["agent"] for m in matched}
    ped_to_agent_patch = dict(ped_to_agent_base)
    ped_to_agent_patch[151] = drop_agents[0]
    ped_to_agent_patch[150] = drop_agents[1]
    for p in drop_peds:
        ped_to_agent_patch.pop(p, None)

    close_peds = [148, 131, 146, 147]
    force_ids = [ped_to_agent_patch[p] for p in close_peds + [151, 150]]
    print("force_ids", force_ids, "->", close_peds + [151, 150])

    model = prepare_model(device)

    base_m, base_c, base_gt, base_logits, base_edges = forward_once(
        model, past_w, future_w, nv, device, force_ids=None
    )
    swap_m, swap_c, swap_gt, swap_logits, swap_edges = forward_once(
        model, past_p, future_p, nv, device, force_ids=None
    )
    force_m, force_c, force_gt, force_logits, force_edges = forward_once(
        model, past_p, future_p, nv, device, force_ids=force_ids
    )

    def traj_delta(a, b):
        return {
            "max_abs_xy": float(np.abs(a - b).max()),
            "mean_abs_xy": float(np.abs(a - b).mean()),
            "best_mode_end_shift_m": float(
                np.linalg.norm(a[base_m["best_ade_mode"] if False else 0] - b[0])
            ),
        }

    # compare force vs baseline using same mode indices
    cand_shift = {
        "force_vs_baseline_max_abs": float(np.abs(force_c - base_c).max()),
        "force_vs_baseline_mean_abs": float(np.abs(force_c - base_c).mean()),
        "force_vs_swap_max_abs": float(np.abs(force_c - swap_c).max()),
        "best_end_shift_force_vs_base_m": float(
            np.linalg.norm(
                np.array(force_m["best_end_world"]) - np.array(base_m["best_end_world"])
            )
        ),
        "top1_end_shift_force_vs_base_m": float(
            np.linalg.norm(
                np.array(force_m["top1_end_world"]) - np.array(base_m["top1_end_world"])
            )
        ),
    }

    delta_fb = {
        k: force_m[k] - base_m[k]
        for k in ("minADE20", "minFDE20", "top1_ADE", "top1_FDE", "oracle_cost")
    }

    # plot 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (axes[0], base_c, base_m, past_w, STORED_PEDS, "baseline default graph"),
        (axes[1], swap_c, swap_m, past_p, new_peds, "swap only (no force)"),
        (axes[2], force_c, force_m, past_p, new_peds, "swap + force edges"),
    ]
    for ax, cands, met, past, peds, title in panels:
        ax.plot(past[0, :, 0], past[0, :, 1], "k-o", ms=3, label="obs")
        ax.plot(base_gt[:, 0], base_gt[:, 1], "g-o", ms=3, label="GT")
        for k in range(len(cands)):
            ax.plot(cands[k, :, 0], cands[k, :, 1], color="0.75", lw=0.5, alpha=0.45)
        best = cands[met["best_ade_mode"]]
        ax.plot(best[:, 0], best[:, 1], "b-", lw=2, label="best ADE")
        t1 = cands[met["top1_mode"]]
        ax.plot(t1[:, 0], t1[:, 1], "r--", lw=1.5, label="top1")
        ax.scatter(past[1:, -1, 0], past[1:, -1, 1], c="orange", s=28, zorder=5)
        for a, ped in enumerate(peds[1:]):
            ax.annotate(str(ped), past[a + 1, -1], fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{title}\nminADE={met['minADE20']:.3f} minFDE={met['minFDE20']:.3f} "
            f"top1={met['top1_ADE']:.3f}"
        )
        ax.legend(fontsize=7)
    fig.suptitle("6802 GRAPH_FORCE: F16 seed1")
    fig.tight_layout()
    plot_path = OUT / "GRAPH_FORCE_compare.png"
    fig.savefig(plot_path, dpi=140)
    plt.close()

    report = {
        "sample_id": SAMPLE_ID,
        "model": "F16 IR1 general seed1",
        "intervention": {
            "neighbor_swap": {"dropped": drop_peds, "inserted": [151, 150], "slots": drop_agents},
            "force_focal_receive_from_agents": force_ids,
            "force_focal_receive_from_peds": close_peds + [151, 150],
            "pad_hack": "forward_hold_first_obs_xy for 151/150 frames 759-762",
            "bypass": "OR-force allowed edges in temporal_social_attention (bypass radius/Top-K for focal receive)",
        },
        "baseline": base_m,
        "swap_only": swap_m,
        "graph_force": force_m,
        "delta_force_minus_baseline": delta_fb,
        "trajectory_shift": cand_shift,
        "edges_baseline": base_edges,
        "edges_swap_only": swap_edges,
        "edges_force": force_edges,
        "plot": str(plot_path),
    }
    (OUT / "GRAPH_FORCE_result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = []
    md.append("# GRAPH_FORCE 6802\n\n")
    md.append("F16 IR1 same weights. Neighbor pack includes 151/150; focal social edges forced to close+BR.\n\n")
    md.append("| condition | minADE20 | minFDE20 | top1 ADE | top1 FDE |\n|---|---|---|---|---|\n")
    for name, m in [("baseline", base_m), ("swap_only", swap_m), ("graph_force", force_m)]:
        md.append(
            f"| {name} | {m['minADE20']:.4f} | {m['minFDE20']:.4f} | {m['top1_ADE']:.4f} | {m['top1_FDE']:.4f} |\n"
        )
    md.append(
        f"\nDelta force−baseline: minADE {delta_fb['minADE20']:+.4f}, minFDE {delta_fb['minFDE20']:+.4f}, "
        f"top1ADE {delta_fb['top1_ADE']:+.4f}, top1FDE {delta_fb['top1_FDE']:+.4f}\n"
    )
    md.append(
        f"\nCand traj max|Δ| force vs baseline: {cand_shift['force_vs_baseline_max_abs']:.6f} m; "
        f"best-end shift {cand_shift['best_end_shift_force_vs_base_m']:.4f} m; "
        f"top1-end shift {cand_shift['top1_end_shift_force_vs_base_m']:.4f} m\n"
    )
    md.append(f"\nForced edges: agents {force_ids} = peds {close_peds + [151, 150]}\n")
    md.append(f"\nPlot: `{plot_path.name}`\n")
    (OUT / "GRAPH_FORCE_NOTE.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps({
        "baseline": {k: base_m[k] for k in ("minADE20", "minFDE20", "top1_ADE", "top1_FDE")},
        "swap_only": {k: swap_m[k] for k in ("minADE20", "minFDE20", "top1_ADE", "top1_FDE")},
        "graph_force": {k: force_m[k] for k in ("minADE20", "minFDE20", "top1_ADE", "top1_FDE")},
        "delta_force_minus_baseline": delta_fb,
        "trajectory_shift": cand_shift,
        "force_edges": force_edges,
    }, indent=2))


if __name__ == "__main__":
    main()
