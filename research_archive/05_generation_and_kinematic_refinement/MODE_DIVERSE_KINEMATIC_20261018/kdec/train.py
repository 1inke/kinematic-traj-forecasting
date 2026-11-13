"""MDK 训练：winner-of-K + 模式间多样性；禁止全员同 GT MSE。"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from . import models as M
from . import metrics as Met

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_MODES = 4


def seed_all(n):
    import random
    random.seed(n); np.random.seed(n); torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def winner_loss(paths, gt):
    e = torch.linalg.vector_norm(paths - gt[:, None], dim=-1)
    return (e.mean(-1) + 0.5 * e[:, :, -1]).min(1).values.mean()


def mode_diversity(paths, n_modes=N_MODES, margin=0.5):
    """鼓励不同 mode 的中心终点远离（不是全员互斥到崩）。"""
    B, K, H, _ = paths.shape
    pm = K // n_modes
    centers = []
    for m in range(n_modes):
        centers.append(paths[:, m*pm:(m+1)*pm, -1].mean(1))  # B,2
    c = torch.stack(centers, 1)  # B,M,2
    d = torch.linalg.vector_norm(c[:, :, None] - c[:, None, :], dim=-1)
    mask = torch.triu(torch.ones(n_modes, n_modes, device=paths.device, dtype=torch.bool), 1)
    return torch.relu(margin - d[:, mask]).mean()


def motion_margin_loss(paths, ref_speed_p95, ref_accel_mean, target_ratio=0.995):
    v = torch.diff(paths, dim=2, prepend=torch.zeros_like(paths[:, :, :1])) / 0.4
    a = torch.diff(v, dim=2) / 0.4
    speed = torch.linalg.vector_norm(v, dim=-1).reshape(-1)
    accel = torch.linalg.vector_norm(a, dim=-1).reshape(-1)
    k = max(1, int(0.05 * speed.numel()))
    sp95 = speed.topk(k).values.mean()
    am = accel.mean()
    return torch.relu(sp95 / ref_speed_p95 - target_ratio) + torch.relu(am / ref_accel_mean - target_ratio)


@torch.no_grad()
def predict(net, past, batch=256):
    net.eval(); rows = []
    for i in range(0, len(past), batch):
        p = torch.from_numpy(past[i:i+batch].astype(np.float32)).to(DEVICE)
        rows.append(net(p)["paths"].cpu().numpy())
    return np.concatenate(rows, 0)


def evaluate_indices(net, data, indices):
    paths = predict(net, data["past"][indices])
    return Met.evaluate_split(paths, data["gt"][indices], data["native"][indices], data["groups"][indices])


def train_arm(data, *, seed=42, epochs=8, batch=256, subset=None,
              div_w=0.1, motion_w=0.05, motion_refs=None, target_ratio=0.995,
              v_max=1.27, a_max=0.57, lr=1e-3, out_dir=None):
    seed_all(seed)
    net = M.build("MDK", v_max=v_max, a_max=a_max).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ix = data["split"]["FIT"]
    if subset is not None:
        ix = ix[:subset]
    history = []; t0 = time.time()
    for ep in range(1, epochs+1):
        net.train()
        order = ix[np.random.permutation(len(ix))]
        rows = []
        for s in range(0, len(order), batch):
            bi = order[s:s+batch]
            past = torch.from_numpy(data["past"][bi].astype(np.float32)).to(DEVICE)
            gt = torch.from_numpy(data["gt"][bi].astype(np.float32)).to(DEVICE)
            out = net(past)
            task = winner_loss(out["paths"], gt)
            div = mode_diversity(out["paths"])
            mot = motion_margin_loss(out["paths"], motion_refs["speed_p95"], motion_refs["accel_mean"], target_ratio) if motion_w > 0 else torch.zeros((), device=DEVICE)
            total = task + div_w * div + motion_w * mot
            opt.zero_grad(set_to_none=True); total.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)); opt.step()
            rows.append(dict(task=float(task.detach()), div=float(div.detach()), motion=float(mot.detach()),
                             total=float(total.detach()), gradnorm=gn, n=len(bi)))
        n = sum(r["n"] for r in rows)
        ep_row = {k: sum(r[k]*r["n"] for r in rows)/n for k in ("task","div","motion","total")}
        ep_row.update(epoch=ep, gradnorm=float(np.mean([r["gradnorm"] for r in rows])))
        history.append(ep_row)
        print(f"[MDK seed={seed}] epoch {ep}/{epochs} " + json.dumps(ep_row), flush=True)
    elapsed = time.time() - t0
    metrics = {
        "FIT": evaluate_indices(net, data, ix if subset is not None else data["split"]["FIT"]),
        "outer": evaluate_indices(net, data, data["split"]["outer"]),
    }
    result = dict(arm="MDK", seed=seed, epochs=epochs, subset=subset, params=M.count_params(net),
                  elapsed_sec=elapsed, history=history, metrics=metrics, device=str(DEVICE))
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "arm": "MDK", "seed": seed, "v_max": v_max, "a_max": a_max},
                   out_dir / f"MDK_seed{seed}.pt")
        (out_dir / f"MDK_seed{seed}_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, net
