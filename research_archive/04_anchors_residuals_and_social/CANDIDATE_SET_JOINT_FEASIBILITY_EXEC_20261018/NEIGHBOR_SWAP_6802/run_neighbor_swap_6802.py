# -*- coding: utf-8 -*-
"""Neighbor-swap counterfactual for sample 6802 (no retrain).

Baseline F16 IR1 checkpoint, original 8 neighbors vs patched:
  drop most-stationary (70, 111), insert BR oncoming 151 (+150 if slot).
Incomplete past of 151/150 (appear frame 763): forward-fill first observed xy
into frames 759-762. Documented hack; model expects full 8-step xy.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
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
VIZ = EXEC / "visual_review_overnight/data/arrays.npz"

SAMPLE_ID = 6802
SEED = 1
SPLIT = "general"
STORED_PEDS = [130, 148, 131, 146, 147, 144, 145, 70, 111]  # agent0=target
OBS_FRAMES = list(range(759, 767))
PRED_FRAMES = list(range(767, 779))
LAMBDA_FDE = 0.5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_label_xy(ped_ids):
    """frame -> ped -> (x,y) from crowds_zara02.txt (cols 0,1,13,15)."""
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
            x = float(parts[13])
            y = float(parts[15])
            table.setdefault(frame, {})[ped] = np.array([x, y], dtype=np.float64)
    return table


def path_length(traj: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())


def build_past_filled(table, ped, frames):
    """8-step past; missing early frames = first observed position (forward-hold)."""
    known = [(f, table[f][ped]) for f in frames if f in table and ped in table[f]]
    if not known:
        raise ValueError(f"ped {ped} missing entirely in obs frames")
    first_xy = known[0][1]
    out = np.zeros((len(frames), 2), dtype=np.float64)
    filled = 0
    for i, f in enumerate(frames):
        if f in table and ped in table[f]:
            out[i] = table[f][ped]
        else:
            out[i] = first_xy
            filled += 1
    return out.astype(np.float32), filled, known[0][0]


def metrics_from_preds(cands: np.ndarray, gt: np.ndarray, logits: np.ndarray | None):
    """cands (K,12,2), gt (12,2)."""
    err = np.linalg.norm(cands - gt[None], axis=-1)
    ade = err.mean(-1)
    fde = err[:, -1]
    cost = ade + LAMBDA_FDE * fde
    out = {
        "minADE20": float(ade.min()),
        "minFDE20": float(fde.min()),
        "oracle_cost": float(cost.min()),
        "best_ade_mode": int(ade.argmin()),
        "best_fde_mode": int(fde.argmin()),
        "best_end_xy": cands[ade.argmin(), -1].tolist(),
        "ade_all": ade.tolist(),
        "fde_all": fde.tolist(),
    }
    if logits is not None:
        top1 = int(np.argmax(logits))
        out["top1_mode"] = top1
        out["top1_ADE"] = float(ade[top1])
        out["top1_FDE"] = float(fde[top1])
        out["top1_end_xy"] = cands[top1, -1].tolist()
    return out


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

    source = SOURCE_ROOT / SPLIT / "f16_temporal_social_cross" / f"seed_{SEED}" / "val_scene_-1" / "best_selected.pth"
    ckpt = IR1 / "results" / SPLIT / "baseline" / "f16_temporal_social_cross" / f"seed_{SEED}" / "val_scene_-1" / "best_selected.pth"
    os.environ["IRS_SOURCE_CHECKPOINT"] = str(source)
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    adapter = load_module("csna_adapter_swap", IR1 / "current_baseline_adapter.py")
    adapter.replace_decoder(model, cfg, "current_mode_anchor_template")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg


@torch.inference_mode()
def forward_once(model, past_xy: np.ndarray, future_xy: np.ndarray, num_valid: int, device):
    """past_xy (A,8,2) world meters; future_xy (A,12,2). Returns focal cands world, logits."""
    from train_front_variant import normalize_to_focal_heading

    past = torch.from_numpy(past_xy[None].astype(np.float32)).to(device)
    future = torch.from_numpy(future_xy[None].astype(np.float32)).to(device)
    nv = torch.tensor([num_valid], dtype=torch.int32, device=device)
    past_n, future_n = normalize_to_focal_heading(past, future)
    center = past_n.mean(dim=-2, keepdim=True)
    pred = model(past_n - center, None, None, mode="testing", num_valid=nv, graph_loc=past_n)[2]
    pred = pred + center.unsqueeze(-3)  # (1,A,K,T,2) heading-normalized frame
    # Invert heading normalize for focal only: normalize_to_focal_heading rotates by heading
    # Model outputs in same frame as past_n. Convert focal preds back to world via same rotation inverse.
    # past_n is heading-normalized world (origin still absolute? check).
    # Looking at CSNA: they compare pred to future after same normalize, so metrics in heading frame
    # are fine for ADE/FDE (rotation-invariant). For world endpoint we invert.
    cands_h = pred[0, 0].cpu().numpy()  # (K,12,2) heading frame
    gt_h = future_n[0, 0].cpu().numpy()
    logits = model.last_mode_logits[0, 0].cpu().numpy()
    # Reconstruct world: heading from original past
    focal = past_xy[0]
    disp = focal[-1] - focal[-4]
    speed = np.linalg.norm(disp)
    if speed > 0.03:
        heading = disp / speed
    else:
        heading = np.array([1.0, 0.0], dtype=np.float64)
    # normalize_to_focal_heading: typically subtract last obs origin then rotate
    # Check implementation
    return cands_h, gt_h, logits, heading, focal[-1].astype(np.float64)


def heading_to_world(local: np.ndarray, origin: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """If local is relative to origin in heading frame (x along heading, y left)."""
    c, s = float(heading[0]), float(heading[1])
    xl, yl = local[..., 0], local[..., 1]
    dx = xl * c - yl * s
    dy = xl * s + yl * c
    return np.stack([dx + origin[0], dy + origin[1]], axis=-1)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    raw = np.load(RAW / "eth_data_train.npy", mmap_mode="r")
    num = np.load(RAW / "eth_num_train.npy")
    nv = int(num[SAMPLE_ID])
    seq = np.array(raw[SAMPLE_ID])  # (9,20,6)
    past_w = seq[:nv, :8, -2:].astype(np.float32)
    future_w = seq[:nv, 8:, -2:].astype(np.float32)

    # Verify ped order by matching end-obs xy to labels
    table = load_label_xy(set(STORED_PEDS + [151, 150, 143]))
    matched = []
    for a in range(nv):
        xy = past_w[a, -1]
        best, bestd = None, 1e9
        for ped in STORED_PEDS:
            if 766 in table and ped in table[766]:
                d = float(np.linalg.norm(xy - table[766][ped]))
                if d < bestd:
                    bestd, best = d, ped
        matched.append({"agent": a, "ped": best, "match_dist": bestd, "path_len": path_length(past_w[a])})
    print("matched", matched)
    assert matched[0]["ped"] == 130

    # Stationary among neighbors (agents 1..): lowest path length
    neigh_stats = sorted(matched[1:], key=lambda r: r["path_len"])
    drop_agents = [neigh_stats[0]["agent"], neigh_stats[1]["agent"]]
    drop_peds = [neigh_stats[0]["ped"], neigh_stats[1]["ped"]]
    print("drop agents", drop_agents, "peds", drop_peds)

    # Build 151 / 150 past with forward-fill
    p151, fill151, first151 = build_past_filled(table, 151, OBS_FRAMES)
    p150, fill150, first150 = build_past_filled(table, 150, OBS_FRAMES)
    print("151 fill_frames", fill151, "first_obs", first151, "past", p151)
    print("150 fill_frames", fill150, "first_obs", first150)

    # Future for new agents (needed for normalize pair shape); from labels
    def build_future(ped):
        out = np.zeros((12, 2), dtype=np.float32)
        for i, f in enumerate(PRED_FRAMES):
            if f in table and ped in table[f]:
                out[i] = table[f][ped]
            else:
                out[i] = out[i - 1] if i else past_w[0, -1]
        return out

    f151, f150 = build_future(151), build_future(150)

    # Patched pack: replace drop slots with 151 and 150
    past_p = past_w.copy()
    future_p = future_w.copy()
    past_p[drop_agents[0]] = p151
    past_p[drop_agents[1]] = p150
    future_p[drop_agents[0]] = f151
    future_p[drop_agents[1]] = f150
    new_peds = STORED_PEDS.copy()
    new_peds[drop_agents[0]] = 151
    new_peds[drop_agents[1]] = 150

    # Overnight cached native reference (world)
    viz = np.load(VIZ)
    row = int(np.where(viz["sample_id"] == SAMPLE_ID)[0][0])
    overnight_native = viz["native"][row]
    overnight_gt = viz["future"][row]
    overnight_metrics = metrics_from_preds(overnight_native, overnight_gt, viz["logits"][row])

    model, cfg = prepare_model(device)

    # Inspect normalize to know world inversion
    from train_front_variant import normalize_to_focal_heading

    past_t = torch.from_numpy(past_w[None])
    fut_t = torch.from_numpy(future_w[None])
    past_n, fut_n = normalize_to_focal_heading(past_t, fut_t)
    # Check: is past_n relative to last obs?
    print("past_w last", past_w[0, -1], "past_n last", past_n[0, 0, -1].numpy())
    print("past_n first", past_n[0, 0, 0].numpy())

    def run(tag, past_xy, future_xy):
        cands_h, gt_h, logits, heading, origin = forward_once(model, past_xy, future_xy, nv, device)
        m = metrics_from_preds(cands_h, gt_h, logits)
        # If past_n last ~ 0, outputs are relative; else absolute in rotated frame
        # Convert endpoints for plot: use same normalize inverse
        # Recompute: get relative world from heading frame assuming origin=last obs and past_n is rotated relative
        rel_w_last = past_w[0] - origin  # should be ~0 at last
        # Build rotation matrix used by normalize
        past_n_np = past_n[0, 0].numpy() if tag == "baseline" else None
        # Safer: metrics already rotation-invariant. For world plot, invert using heading on *relative* preds.
        # Detect if cands are relative: compare scale
        cands_rel = cands_h.copy()
        # Heuristic from CSNA: center subtracted before model, added back -> absolute in heading-norm frame (still rotated world coords, not relative to 0)
        # past_n last for focal:
        pn = normalize_to_focal_heading(
            torch.from_numpy(past_xy[None].astype(np.float32)),
            torch.from_numpy(future_xy[None].astype(np.float32)),
        )[0][0, 0].numpy()
        # If pn[-1] is near origin of heading frame at last obs position rotated...
        # Inverse rotation only (no translation) if absolute rotated:
        c, s = float(heading[0]), float(heading[1])
        # R = [[c,s],[-s,c]] maps world-rel to heading; inverse [[c,-s],[s,c]]
        def inv_rot(p):
            xl, yl = p[..., 0], p[..., 1]
            return np.stack([xl * c - yl * s, xl * s + yl * c], axis=-1)

        # Check: inv_rot(pn) + ? == past_xy[0]
        inv_pn = inv_rot(pn)
        # If normalize only rotates about last obs after subtracting origin:
        # pn = R @ (p - origin) => p = R^{-1}@pn + origin
        recon = inv_rot(pn) + origin
        recon_err = float(np.abs(recon - past_xy[0]).max())
        print(tag, "recon_err", recon_err)
        cands_w = inv_rot(cands_h) + origin
        gt_w = inv_rot(gt_h) + origin
        m["end_gt_world"] = gt_w[-1].tolist()
        m["best_end_world"] = cands_w[m["best_ade_mode"], -1].tolist()
        if "top1_mode" in m:
            m["top1_end_world"] = cands_w[m["top1_mode"], -1].tolist()
        return {
            "tag": tag,
            "metrics": m,
            "cands_world": cands_w,
            "gt_world": gt_w,
            "logits": logits,
            "recon_err": recon_err,
            "heading": heading.tolist(),
            "origin": origin.tolist(),
        }

    base = run("baseline", past_w, future_w)
    patch = run("patched", past_p, future_p)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res, title in [
        (axes[0], base, "baseline neighbors"),
        (axes[1], patch, "patched: +151,+150 -stat"),
    ]:
        gt = res["gt_world"]
        ax.plot(past_w[0, :, 0], past_w[0, :, 1], "k-o", ms=3, label="obs target")
        ax.plot(gt[:, 0], gt[:, 1], "g-o", ms=3, label="GT future")
        cands = res["cands_world"]
        for k in range(len(cands)):
            ax.plot(cands[k, :, 0], cands[k, :, 1], color="0.7", lw=0.6, alpha=0.5)
        best = cands[res["metrics"]["best_ade_mode"]]
        ax.plot(best[:, 0], best[:, 1], "b-", lw=2, label="best ADE mode")
        if "top1_mode" in res["metrics"]:
            t1 = cands[res["metrics"]["top1_mode"]]
            ax.plot(t1[:, 0], t1[:, 1], "r--", lw=1.5, label="top1 logit")
        # neighbors last pos
        if res["tag"] == "baseline":
            nb = past_w[1:]
            labs = STORED_PEDS[1:]
        else:
            nb = past_p[1:]
            labs = new_peds[1:]
        ax.scatter(nb[:, -1, 0], nb[:, -1, 1], c="orange", s=30, zorder=5)
        for a, ped in enumerate(labs):
            ax.annotate(str(ped), nb[a, -1], fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{title}\nminADE={res['metrics']['minADE20']:.3f} minFDE={res['metrics']['minFDE20']:.3f} "
            f"top1ADE={res['metrics'].get('top1_ADE', float('nan')):.3f}"
        )
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("6802 neighbor-swap F16 seed1 (no retrain)")
    fig.tight_layout()
    plot_path = OUT / "compare_6802_neighbor_swap.png"
    fig.savefig(plot_path, dpi=140)
    plt.close()

    delta = {
        "minADE20": patch["metrics"]["minADE20"] - base["metrics"]["minADE20"],
        "minFDE20": patch["metrics"]["minFDE20"] - base["metrics"]["minFDE20"],
        "top1_ADE": patch["metrics"]["top1_ADE"] - base["metrics"]["top1_ADE"],
        "top1_FDE": patch["metrics"]["top1_FDE"] - base["metrics"]["top1_FDE"],
        "oracle_cost": patch["metrics"]["oracle_cost"] - base["metrics"]["oracle_cost"],
    }

    report = {
        "sample_id": SAMPLE_ID,
        "model": {
            "name": "F16 IR1 baseline current_mode_anchor_template",
            "split": SPLIT,
            "seed": SEED,
            "checkpoint": str(
                IR1
                / "results"
                / SPLIT
                / "baseline"
                / "f16_temporal_social_cross"
                / f"seed_{SEED}"
                / "val_scene_-1"
                / "best_selected.pth"
            ),
            "note": "Not regenerating overnight native/k1_S pool; those caches bake original neighbors. This is controlled encoder+decoder re-inference.",
        },
        "original_neighbors": STORED_PEDS[1:],
        "patched_neighbors": new_peds[1:],
        "dropped": [{"agent": a, "ped": p, "path_len": path_length(past_w[a])} for a, p in zip(drop_agents, drop_peds)],
        "inserted": [
            {
                "ped": 151,
                "slot_agent": drop_agents[0],
                "fill_frames_759_762": fill151,
                "pad": "forward_hold_first_observed_xy",
                "first_obs_frame": first151,
                "dist_at_766_m": float(np.linalg.norm(p151[-1] - past_w[0, -1])),
            },
            {
                "ped": 150,
                "slot_agent": drop_agents[1],
                "fill_frames_759_762": fill150,
                "pad": "forward_hold_first_observed_xy",
                "first_obs_frame": first150,
                "dist_at_766_m": float(np.linalg.norm(p150[-1] - past_w[0, -1])),
            },
        ],
        "baseline": base["metrics"],
        "patched": patch["metrics"],
        "delta_patched_minus_baseline": delta,
        "overnight_native_reference_world": overnight_metrics,
        "recon_err_baseline": base["recon_err"],
        "recon_err_patched": patch["recon_err"],
        "plot": str(plot_path),
    }

    (OUT / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = []
    md.append("# Neighbor swap 6802 (no retrain)\n")
    md.append(f"- Model: F16 IR1 `{SPLIT}` seed{SEED} (encoder+decoder forward only)\n")
    md.append(f"- Drop stationary: ped **{drop_peds}** (path_len ~0)\n")
    md.append(f"- Insert BR: **151, 150** (pad frames 759–762 = hold first xy @763)\n")
    md.append("\n## Metrics (heading-normalized frame; rotation-invariant ADE/FDE)\n\n")
    md.append("| | minADE20 | minFDE20 | top1 ADE | top1 FDE |\n|---|---|---|---|---|\n")
    md.append(
        f"| baseline | {base['metrics']['minADE20']:.4f} | {base['metrics']['minFDE20']:.4f} | "
        f"{base['metrics']['top1_ADE']:.4f} | {base['metrics']['top1_FDE']:.4f} |\n"
    )
    md.append(
        f"| patched | {patch['metrics']['minADE20']:.4f} | {patch['metrics']['minFDE20']:.4f} | "
        f"{patch['metrics']['top1_ADE']:.4f} | {patch['metrics']['top1_FDE']:.4f} |\n"
    )
    md.append(
        f"| delta (p-b) | {delta['minADE20']:+.4f} | {delta['minFDE20']:+.4f} | "
        f"{delta['top1_ADE']:+.4f} | {delta['top1_FDE']:+.4f} |\n"
    )
    md.append(
        f"\nOvernight cached native (original neighbors, world): minADE={overnight_metrics['minADE20']:.4f} "
        f"minFDE={overnight_metrics['minFDE20']:.4f} top1ADE={overnight_metrics['top1_ADE']:.4f}\n"
    )
    md.append(f"\nPlot: `{plot_path.name}`\n")
    (OUT / "NOTE.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({"delta": delta, "baseline": base["metrics"], "patched": patch["metrics"]}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
