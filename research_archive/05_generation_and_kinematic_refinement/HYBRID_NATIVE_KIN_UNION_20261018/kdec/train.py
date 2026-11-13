"""训练与评估。"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from . import models as M
from . import metrics as Met

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_all(n: int):
    import random
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def winner_loss(paths, gt):
    e = torch.linalg.vector_norm(paths - gt[:, None], dim=-1)
    return (e.mean(-1) + 0.5 * e[:, :, -1]).min(1).values.mean()


def diversity_hinge(paths, margin=0.3):
    end = paths[:, :, -1]
    d = torch.linalg.vector_norm(end[:, :, None] - end[:, None, :], dim=-1)
    K = paths.shape[1]
    mask = torch.triu(torch.ones(K, K, device=paths.device, dtype=torch.bool), 1)
    return torch.relu(margin - d[:, mask]).mean()


def consistency_loss(out):
    if "v_hat" not in out:
        return torch.zeros((), device=out["paths"].device)
    return (
        torch.linalg.vector_norm(out["v_hat"] - out["v_int"], dim=-1).mean()
        + torch.linalg.vector_norm(out["dp_hat"] - out["dp_int"], dim=-1).mean()
    )


def motion_margin_loss(paths, ref_speed_p95, ref_accel_mean, target_ratio=0.995):
    """可微近似：batch 内 speed 分位与 accel 均值相对参考的超标惩罚。"""
    v = torch.diff(paths, dim=2, prepend=torch.zeros_like(paths[:, :, :1])) / 0.4
    a = torch.diff(v, dim=2) / 0.4
    speed = torch.linalg.vector_norm(v, dim=-1).reshape(-1)
    accel = torch.linalg.vector_norm(a, dim=-1).reshape(-1)
    # soft quantile via topk
    k = max(1, int(0.05 * speed.numel()))
    sp95 = speed.topk(k).values.mean()
    am = accel.mean()
    return (
        torch.relu(sp95 / ref_speed_p95 - target_ratio)
        + torch.relu(am / ref_accel_mean - target_ratio)
    )


def batch_loss(out, gt, *, cons_w=0.1, div_w=0.05, motion_w=0.0, motion_refs=None, target_ratio=0.995):
    task = winner_loss(out["paths"], gt)
    div = diversity_hinge(out["paths"])
    cons = consistency_loss(out)
    mot = torch.zeros((), device=out["paths"].device)
    if motion_w > 0 and motion_refs is not None and "v_hat" in out:
        mot = motion_margin_loss(out["paths"], motion_refs["speed_p95"], motion_refs["accel_mean"], target_ratio)
    total = task + div_w * div + cons_w * cons + motion_w * mot
    return total, dict(
        task=float(task.detach()),
        div=float(div.detach()),
        cons=float(cons.detach()),
        motion=float(mot.detach()),
        total=float(total.detach()),
    )


@torch.no_grad()
def predict(net, past, batch=256):
    net.eval()
    rows = []
    for i in range(0, len(past), batch):
        p = torch.from_numpy(past[i : i + batch].astype(np.float32)).to(DEVICE)
        rows.append(net(p)["paths"].detach().cpu().numpy())
    return np.concatenate(rows, 0)


def evaluate_indices(net, data, indices):
    paths = predict(net, data["past"][indices])
    return Met.evaluate_split(paths, data["gt"][indices], data["native"][indices], data["groups"][indices])


def train_arm(
    arm,
    data,
    *,
    seed=42,
    epochs=8,
    batch=256,
    subset=None,
    cons_w=0.1,
    div_w=0.05,
    motion_w=0.0,
    motion_refs=None,
    target_ratio=0.995,
    v_max=1.27,
    a_max=0.57,
    lr=1e-3,
    out_dir: Path | None = None,
):
    seed_all(seed)
    net = M.build(arm, v_max=v_max, a_max=a_max).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ix = data["split"]["FIT"]
    if subset is not None:
        ix = ix[:subset]
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        order = ix[np.random.permutation(len(ix))]
        rows = []
        for s in range(0, len(order), batch):
            bi = order[s : s + batch]
            past = torch.from_numpy(data["past"][bi].astype(np.float32)).to(DEVICE)
            gt = torch.from_numpy(data["gt"][bi].astype(np.float32)).to(DEVICE)
            out = net(past)
            mw = motion_w if arm == "KIN" else 0.0
            total, parts = batch_loss(
                out, gt, cons_w=cons_w, div_w=div_w, motion_w=mw,
                motion_refs=motion_refs, target_ratio=target_ratio,
            )
            opt.zero_grad(set_to_none=True)
            total.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0))
            opt.step()
            parts["gradnorm"] = gn
            parts["n"] = len(bi)
            rows.append(parts)
        n = sum(r["n"] for r in rows)
        ep_row = {
            "epoch": ep,
            **{k: sum(r[k] * r["n"] for r in rows) / n for k in ("task", "div", "cons", "motion", "total")},
            "gradnorm": float(np.mean([r["gradnorm"] for r in rows])),
        }
        history.append(ep_row)
        print(f"[{arm} seed={seed}] epoch {ep}/{epochs} " + json.dumps(ep_row), flush=True)
    elapsed = time.time() - t0
    metrics = {
        "FIT": evaluate_indices(net, data, ix if subset is not None else data["split"]["FIT"]),
        "outer": evaluate_indices(net, data, data["split"]["outer"]),
    }
    result = dict(
        arm=arm, seed=seed, epochs=epochs, subset=subset, params=M.count_params(net),
        elapsed_sec=elapsed, history=history, metrics=metrics, device=str(DEVICE),
        v_max=v_max, a_max=a_max, motion_w=motion_w if arm == "KIN" else 0.0,
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / f"{arm}_seed{seed}.pt"
        torch.save({"state_dict": net.state_dict(), "arm": arm, "seed": seed, "v_max": v_max, "a_max": a_max}, ckpt)
        (out_dir / f"{arm}_seed{seed}_RESULT.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["checkpoint"] = str(ckpt)
    return result, net
