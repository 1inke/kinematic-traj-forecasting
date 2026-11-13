"""训练与评估入口。"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from . import data as D
from . import models as M
from . import metrics as Met
from .integrate import DT

HERE = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_all(n: int):
    import random
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def winner_loss(paths: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """min_k (ADE + 0.5 FDE). paths(B,K,H,2) gt(B,H,2)"""
    e = torch.linalg.vector_norm(paths - gt[:, None], dim=-1)
    ade = e.mean(-1)
    fde = e[:, :, -1]
    score = ade + 0.5 * fde
    return score.min(1).values.mean()


def diversity_hinge(paths: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """轻多样性：终点两两距离。"""
    end = paths[:, :, -1]  # B,K,2
    d = torch.linalg.vector_norm(end[:, :, None] - end[:, None, :], dim=-1)
    # upper triangle
    K = paths.shape[1]
    mask = torch.triu(torch.ones(K, K, device=paths.device, dtype=torch.bool), 1)
    pair = d[:, mask]
    return torch.relu(margin - pair).mean()


def consistency_loss(out: dict) -> torch.Tensor:
    if "v_hat" not in out:
        return torch.zeros((), device=out["paths"].device)
    return (
        torch.linalg.vector_norm(out["v_hat"] - out["v_int"], dim=-1).mean()
        + torch.linalg.vector_norm(out["dp_hat"] - out["dp_int"], dim=-1).mean()
    )


def batch_loss(out: dict, gt: torch.Tensor, cons_w: float = 0.1, div_w: float = 0.05):
    task = winner_loss(out["paths"], gt)
    div = diversity_hinge(out["paths"])
    cons = consistency_loss(out)
    total = task + div_w * div + cons_w * cons
    return total, dict(task=float(task.detach()), div=float(div.detach()), cons=float(cons.detach()), total=float(total.detach()))


@torch.no_grad()
def predict(net: nn.Module, past: np.ndarray, batch: int = 256) -> np.ndarray:
    net.eval()
    rows = []
    for i in range(0, len(past), batch):
        p = torch.from_numpy(past[i : i + batch].astype(np.float32)).to(DEVICE)
        out = net(p)
        rows.append(out["paths"].detach().cpu().numpy())
    return np.concatenate(rows, 0)


def evaluate_indices(net, data, indices):
    paths = predict(net, data["past"][indices])
    return Met.evaluate_split(
        paths,
        data["gt"][indices],
        data["native"][indices],
        data["groups"][indices],
    )


def train_arm(
    arm: str,
    data: dict,
    *,
    seed: int = 42,
    epochs: int = 8,
    batch: int = 256,
    subset: int | None = None,
    cons_w: float = 0.1,
    lr: float = 1e-3,
    out_dir: Path | None = None,
):
    seed_all(seed)
    net = M.build(arm).to(DEVICE)
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
            total, parts = batch_loss(out, gt, cons_w=cons_w)
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
            "task": sum(r["task"] * r["n"] for r in rows) / n,
            "div": sum(r["div"] * r["n"] for r in rows) / n,
            "cons": sum(r["cons"] * r["n"] for r in rows) / n,
            "total": sum(r["total"] * r["n"] for r in rows) / n,
            "gradnorm": float(np.mean([r["gradnorm"] for r in rows])),
        }
        history.append(ep_row)
        print(f"[{arm}] epoch {ep}/{epochs} " + json.dumps(ep_row), flush=True)
    elapsed = time.time() - t0
    metrics = {
        "FIT": evaluate_indices(net, data, data["split"]["FIT"] if subset is None else ix),
        "outer": evaluate_indices(net, data, data["split"]["outer"]),
    }
    result = dict(
        arm=arm,
        seed=seed,
        epochs=epochs,
        subset=subset,
        params=M.count_params(net),
        elapsed_sec=elapsed,
        history=history,
        metrics=metrics,
        device=str(DEVICE),
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / f"{arm}_seed{seed}.pt"
        torch.save({"state_dict": net.state_dict(), "arm": arm, "seed": seed}, ckpt)
        (out_dir / f"{arm}_seed{seed}_RESULT.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["checkpoint"] = str(ckpt)
    return result, net
