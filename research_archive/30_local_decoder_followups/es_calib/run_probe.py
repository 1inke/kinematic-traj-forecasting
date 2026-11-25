"""Fast, zero-leakage end-to-end pilot on official `univ` fold (64 epochs, seed 42).

Tests whether 1-layer causal refinement + differentiable Energy Score loss
(removing 1-hot sigmoid BCE and logit-sharpening hinge) fixes stationary
duplicate collapse and hard-case overconfidence while preserving accuracy gains.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"c:\Users\linke\Documents\New project")
WORKSPACE = ROOT / "qoder_work/EXP_QRANK_MLDEC_L4_EXEC_20261122/local/workspace"
QROOT = ROOT / "qoder_work/EXP_QRANK_MLDEC_L4_EXEC_20261122"
REV_RUNS = QROOT / "review/formal_20261123/stage2/runs"
OUT_ROOT = ROOT / "scratch/probe_es_calib_20261124"

sys.path.insert(0, str(WORKSPACE))
from advisor_train import core, inputs, load_data  # noqa: E402
from advisor_model import attach as attach_advisor  # noqa: E402
from advisor_eval import forecast_metrics, groups  # noqa: E402
from order_numpy import conf_order, risk_order, softmax  # noqa: E402
from qrank_eval import infer as qrank_infer  # noqa: E402
from qrank_model import attach as attach_qrank  # noqa: E402


DT = 0.4


def energy_score_loss(pred: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Differentiable 24D Energy Score loss: E_p||X - y|| - 0.5 E_{p,p}||X - X'||."""
    X = pred[:, 0].reshape(pred.size(0), pred.size(2), -1)  # (B, K, 24)
    y = target[:, 0].unsqueeze(1).reshape(target.size(0), 1, -1)  # (B, 1, 24)
    p = torch.softmax(logits[:, 0], dim=-1)  # (B, K)
    d_xy = torch.sqrt(((X - y) ** 2).sum(dim=-1) + eps * eps)
    term1 = (p * d_xy).sum(dim=-1)
    diff = X.unsqueeze(2) - X.unsqueeze(1)  # (B, K, K, 24)
    eye = torch.eye(X.size(1), device=X.device, dtype=torch.bool).unsqueeze(0)
    d_xx = torch.sqrt((diff ** 2).sum(dim=-1) + eps * eps).masked_fill(eye, 0.0)
    term2 = 0.5 * torch.einsum("bi,bj,bij->b", p, p, d_xx)
    return (term1 - term2).mean()


def kde_nll_vec(paths: np.ndarray, gt: np.ndarray, weights: np.ndarray | None = None, log_pdf_lower_bound: float = -20.0) -> np.ndarray:
    """Exact vectorized Trajectron++ 2D Gaussian KDE-NLL (Scott's rule, dt-wise mean)."""
    N, K, T, _ = paths.shape
    if weights is None:
        w = np.full((N, K), 1.0 / K, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w, axis=1, keepdims=True)
    sum_w2 = np.sum(w * w, axis=1)
    neff = 1.0 / np.maximum(sum_w2, 1e-12)
    factor2 = np.power(neff, -2.0 / 6.0)
    denom = np.maximum(1.0 - sum_w2, 1e-12)

    w_exp = w[:, :, None, None]
    X = paths.astype(np.float64)
    mu = np.sum(w_exp * X, axis=1, keepdims=True)
    dx = X - mu
    a = np.sum(w_exp[..., 0] * dx[..., 0] * dx[..., 0], axis=1) / denom[:, None]
    b = np.sum(w_exp[..., 0] * dx[..., 0] * dx[..., 1], axis=1) / denom[:, None]
    c = np.sum(w_exp[..., 0] * dx[..., 1] * dx[..., 1], axis=1) / denom[:, None]
    f2 = factor2[:, None]
    sa, sb, sc = a * f2, b * f2, c * f2
    det = sa * sc - sb * sb
    bad = det <= 1e-18
    if bad.any():
        sa = np.where(bad, sa + 1e-8, sa)
        sc = np.where(bad, sc + 1e-8, sc)
        det = sa * sc - sb * sb
    inv_a, inv_b, inv_c = sc / det, -sb / det, sa / det
    log_norm = -np.log(2.0 * np.pi) - 0.5 * np.log(det)

    diff = gt[:, None, :, :].astype(np.float64) - X
    u, v = diff[..., 0], diff[..., 1]
    quad = inv_a[:, None, :] * u * u + 2.0 * inv_b[:, None, :] * u * v + inv_c[:, None, :] * v * v
    log_w = np.log(np.maximum(w, 1e-300))[:, :, None]
    log_terms = log_w - 0.5 * quad
    m = np.max(log_terms, axis=1, keepdims=True)
    log_pdf = log_norm + m[:, 0, :] + np.log(np.sum(np.exp(log_terms - m), axis=1))
    return -np.maximum(log_pdf, log_pdf_lower_bound).mean(axis=1)


def compute_es_numpy(paths: np.ndarray, logits: np.ndarray, gt: np.ndarray, weighted: bool = True) -> np.ndarray:
    """24D Energy Score matching advisor_eval.forecast_metrics."""
    N, K, T, _ = paths.shape
    X = paths.reshape(N, K, T * 2).astype(np.float64)
    y = gt.reshape(N, 1, T * 2).astype(np.float64)
    w = softmax(logits.astype(np.float64)) if weighted else np.full((N, K), 1.0 / K, dtype=np.float64)
    d_xy = np.linalg.norm(X - y, axis=-1)
    term1 = (w * d_xy).sum(axis=1)
    i_idx, j_idx = np.triu_indices(K, 1)
    scatter = np.empty(N, dtype=np.float64)
    for lo in range(0, N, 512):
        sl = slice(lo, lo + 512)
        d = np.linalg.norm(X[sl, i_idx] - X[sl, j_idx], axis=-1)
        scatter[sl] = (d * w[sl, i_idx] * w[sl, j_idx]).sum(axis=1)
    return term1 - scatter


def motion_stats(paths: np.ndarray, past_focal: np.ndarray) -> dict[str, float]:
    """All-K20 pooled kinematics (dt=0.4s, prepending last observed point for speed/accel/jerk + junction)."""
    p = np.broadcast_to(past_focal[:, None], paths.shape[:2] + past_focal.shape[1:])
    v = np.diff(np.concatenate([p[..., -1:, :], paths], axis=-2), axis=-2) / DT
    a = np.diff(v, axis=-2) / DT
    j = np.diff(a, axis=-2) / DT
    pv = (p[..., -1, :] - p[..., -2, :]) / DT
    pa = (p[..., -1, :] - 2 * p[..., -2, :] + p[..., -3, :]) / (DT ** 2)
    ja = (v[..., 0, :] - pv) / DT
    jj = (ja - pa) / DT
    sp = np.linalg.norm(v, axis=-1)
    ac = np.linalg.norm(a, axis=-1)
    jk = np.linalg.norm(j, axis=-1)
    jac = np.linalg.norm(ja, axis=-1)
    jjk = np.linalg.norm(jj, axis=-1)
    return {
        "speed_p95": float(np.quantile(sp, 0.95)),
        "accel_mean": float(ac.mean()),
        "accel_p95": float(np.quantile(ac, 0.95)),
        "jerk_mean": float(jk.mean()),
        "jerk_p95": float(np.quantile(jk, 0.95)),
        "junction_accel_mean": float(jac.mean()),
        "junction_accel_p95": float(np.quantile(jac, 0.95)),
        "junction_jerk_mean": float(jjk.mean()),
        "junction_jerk_p95": float(np.quantile(jjk, 0.95)),
    }


class ProbeTrainer(core.Trainer):
    """Exact core.Trainer + 1-layer S2 causal attachment + optional Energy Score loss."""

    def __init__(self, data, seed: int, device: str, epochs: int, lambda_es: float):
        super().__init__(data, np.arange(len(data["ids"])), batch=64, epochs=epochs,
                         device=device, seed=seed, order_seed=seed, arm="B0")
        self.lambda_es = float(lambda_es)
        attach_advisor(self.model, "S2")
        self.ema = core.ModelEMA(self.model, 0.995)
        self.optimizer, self.scheduler = core.optimizer_scheduler(self.model, self.steps_per_epoch, epochs)

    def step(self):
        if self.order is None or self.cursor == len(self.order):
            if self.epoch >= self.epochs:
                raise StopIteration
            self.order = self.indices[torch.randperm(len(self.indices), generator=self.generator).numpy()]
            self.cursor, self.epoch = 0, self.epoch + 1
        idx = self.order[self.cursor:self.cursor + self.batch]
        self.model.train()
        past, future, tokens = (torch.as_tensor(self.data[k][idx], device=self.device)
                                for k in ("past", "future", "num_valid"))
        pred, logits, target, _ = core.prediction(self.model, past, tokens, future, augment=True)
        aux_p, aux_z = self.model.decoder.aux
        l0, _ = core.loss_value(aux_p, target, aux_z, self.loss_fn)
        l1_base, detail = core.loss_value(pred, target, logits, self.loss_fn)
        l_es = energy_score_loss(pred, logits, target)
        loss = 0.5 * l0 + l1_base + self.lambda_es * l_es
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss: {loss.item()}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            5.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.ema.update(self.model)
        self.scheduler.step()
        self.updates += 1
        self.cursor += len(idx)
        return dict(
            loss=float(loss.detach()),
            l0=float(l0.detach()),
            l1_base=float(l1_base.detach()),
            l_es=float(l_es.detach()),
            reg=float(detail["regression"].detach()),
            cls=float(detail["classification"].detach()),
            grad_norm=float(grad),
        )


@torch.no_grad()
def infer_1layer(model, data, batch: int = 64):
    model.eval()
    device = next(model.parameters()).device
    all_p, all_z = [], []
    for lo in range(0, len(data["ids"]), batch):
        sl = slice(lo, lo + batch)
        p, n = (torch.as_tensor(data[k][sl], device=device) for k in ("past", "num_valid"))
        out, z, _, tf = core.prediction(model, p, n)
        all_p.append(core.world_paths(out, tf).double().cpu().numpy())
        all_z.append(z[:, 0].double().cpu().numpy())
    return np.concatenate(all_p, axis=0), np.concatenate(all_z, axis=0)


@torch.no_grad()
def infer_b0(model, data):
    pred = qrank_infer(core, model, data)
    lp, ll = pred["layer_paths"], pred["layer_logits"]
    paths = lp[:, 0] if lp.ndim == 5 else lp
    logits = ll[:, 0] if ll.ndim == 3 else ll
    return paths.astype(np.float64), logits.astype(np.float64)


@torch.no_grad()
def infer_qrank(model, data):
    pred = qrank_infer(core, model, data)
    return pred["layer_paths"][:, 3].astype(np.float64), pred["layer_logits"][:, 3].astype(np.float64)


def run_smoke(fit_data, device: str):
    print("=== SMOKE & TIMING PRECHECK ===", flush=True)
    # 1. Verify differentiable ES loss matches NumPy exact ES on a batch
    smoke_fit = {k: v[:128].copy() for k, v in fit_data.items()}
    trainer = ProbeTrainer(smoke_fit, seed=42, device=device, epochs=2, lambda_es=0.25)
    trainer.model.train()
    past = torch.as_tensor(smoke_fit["past"][:64], device=device)
    future = torch.as_tensor(smoke_fit["future"][:64], device=device)
    tokens = torch.as_tensor(smoke_fit["num_valid"][:64], device=device)
    pred, logits, target, _ = core.prediction(trainer.model, past, tokens, future, augment=False)
    es_torch = float(energy_score_loss(pred, logits, target).detach().cpu())
    es_np = float(compute_es_numpy(pred[:, 0].detach().cpu().numpy(),
                                   logits[:, 0].detach().cpu().numpy(),
                                   target[:, 0].detach().cpu().numpy(),
                                   weighted=True).mean())
    es_diff = abs(es_torch - es_np)
    print(f"[SMOKE-ES] torch={es_torch:.6f}, numpy={es_np:.6f}, abs_diff={es_diff:.3e}", flush=True)
    assert es_diff < 1e-4, f"ES mismatch: {es_diff}"

    # 2. 2-step gradient & finiteness check
    for step_idx in range(2):
        # run step manually once to inspect gradients before zeroing
        if step_idx == 0:
            row = trainer.step()
            print(f"[SMOKE-STEP 1] {row}", flush=True)
        else:
            # inspect offset and score grads directly
            idx = np.arange(64, 128)
            p2 = torch.as_tensor(smoke_fit["past"][idx], device=device)
            f2 = torch.as_tensor(smoke_fit["future"][idx], device=device)
            n2 = torch.as_tensor(smoke_fit["num_valid"][idx], device=device)
            pr, lg, tg, _ = core.prediction(trainer.model, p2, n2, f2, augment=True)
            ap, az = trainer.model.decoder.aux
            loss = (0.5 * core.loss_value(ap, tg, az, trainer.loss_fn)[0]
                    + core.loss_value(pr, tg, lg, trainer.loss_fn)[0]
                    + 0.25 * energy_score_loss(pr, lg, tg))
            assert torch.isfinite(loss)
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            off_grad = max(p.grad.abs().max().item() for p in trainer.model.decoder.offset.parameters())
            scr_grad = max(p.grad.abs().max().item() for p in trainer.model.decoder.score.parameters())
            print(f"[SMOKE-STEP 2] offset_max_grad={off_grad:.3e}, score_max_grad={scr_grad:.3e}", flush=True)
            assert off_grad > 0.0 and scr_grad > 0.0, "Zero gradient on offset or score!"

    # 3. 1-epoch univ GPU timing benchmark
    t0 = time.monotonic()
    bench_trainer = ProbeTrainer(fit_data, seed=42, device=device, epochs=64, lambda_es=0.25)
    t_ep0 = time.monotonic()
    for _ in range(bench_trainer.steps_per_epoch):
        row = bench_trainer.step()
    ep_sec = time.monotonic() - t_ep0
    print(f"[TIMING] univ 1 epoch ({bench_trainer.steps_per_epoch} steps) = {ep_sec:.2f}s "
          f"(est 64 epochs = {ep_sec*64/60:.1f} min, last_loss={row['loss']:.4f})", flush=True)
    if ep_sec > 45.0:
        raise RuntimeError(f"1-epoch time {ep_sec:.2f}s > 45s budget; stopping for re-plan.")
    return ep_sec


def evaluate_predictions(name: str, paths: np.ndarray, logits: np.ndarray, data: dict,
                         masks: dict, n_total: int, n_train: int,
                         b0_motion: dict[str, float] | None = None) -> dict:
    gt = data["future"].astype(np.float64)
    paths = paths.astype(np.float64)
    logits = logits.astype(np.float64)
    N = len(gt)
    row = np.arange(N)
    p = softmax(logits)

    fm = forecast_metrics(paths, logits, gt)
    err = np.linalg.norm(paths - gt[:, None], axis=-1)
    ade = err.mean(axis=-1)
    fde = err[:, :, -1]
    best_i = np.argmin(fde, axis=1)
    p_star = p[row, best_i]
    b_pen = (1.0 - p_star) ** 2

    ep = paths[:, :, -1, :]
    near_best = np.linalg.norm(ep - ep[row, best_i][:, None, :], axis=-1) <= 0.2
    p_near = (p * near_best).sum(axis=1)
    n_near = near_best.sum(axis=1).astype(np.float64)

    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    maxp = p.max(axis=1)

    es_w = fm["ES"]
    es_u = compute_es_numpy(paths, logits, gt, weighted=False)
    kde = kde_nll_vec(paths, gt, weights=None)
    wkde = kde_nll_vec(paths, gt, weights=p)

    # Top-K ranking (confidence vs risk)
    c_ord = conf_order(logits)
    r_ord = risk_order(ep, p)
    c_fde = np.take_along_axis(fde, c_ord, axis=1)
    r_fde = np.take_along_axis(fde, r_ord, axis=1)

    st = masks["stationary"].astype(bool)
    hd = masks["hard"].astype(bool)

    mot = motion_stats(paths, data["past"][:, 0].astype(np.float64))
    mot_rel = {}
    if b0_motion is not None:
        for k, v in mot.items():
            mot_rel[k] = (v - b0_motion[k]) / b0_motion[k] * 100.0

    summary = dict(
        name=name,
        total_params=int(n_total),
        trainable_params=int(n_train),
        minADE20=float(fm["minADE"].mean()),
        minFDE20=float(fm["minFDE"].mean()),
        J_cov_pct=float(fm["J"].mean() * 100.0),
        brier_minFDE=float(fm["brier_minFDE"].mean()),
        B_pen=float(b_pen.mean()),
        p_star=float(p_star.mean()),
        H_p=float(ent.mean()),
        maxP=float(maxp.mean()),
        p_near_02=float(p_near.mean()),
        n_near_02=float(n_near.mean()),
        st_p_near_02=float(p_near[st].mean()),
        st_n_near_02=float(n_near[st].mean()),
        st_p_star=float(p_star[st].mean()),
        st_H_p=float(ent[st].mean()),
        st_maxP=float(maxp[st].mean()),
        ES_w=float(es_w.mean()),
        ES_u=float(es_u.mean()),
        KDE_NLL=float(kde.mean()),
        wKDE_NLL=float(wkde.mean()),
        hard_ES_w=float(es_w[hd].mean()),
        hard_ES_u=float(es_u[hd].mean()),
        hard_KDE_NLL=float(kde[hd].mean()),
        hard_wKDE_NLL=float(wkde[hd].mean()),
        Top1_FDE=float(fm["Top1_FDE"].mean()),
        cFDE3=float(c_fde[:, :3].min(1).mean()),
        rFDE3=float(r_fde[:, :3].min(1).mean()),
        cFDE5=float(c_fde[:, :5].min(1).mean()),
        rFDE5=float(r_fde[:, :5].min(1).mean()),
        motion=mot,
        motion_vs_B0_pct=mot_rel,
    )
    return summary


def train_arm(arm_name: str, lambda_es: float, fit_data, test_data,
              epochs: int, seed: int, device: str):
    out_dir = OUT_ROOT / f"{arm_name}_univ_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = ProbeTrainer(fit_data, seed=seed, device=device, epochs=epochs, lambda_es=lambda_es)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n=== TRAIN {arm_name} (lambda_es={lambda_es}, epochs={epochs}, params={n_total}) ===", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.monotonic()
    history = []
    for ep in range(1, epochs + 1):
        t_ep = time.monotonic()
        rows = [trainer.step() for _ in range(trainer.steps_per_epoch)]
        dt = time.monotonic() - t_ep
        mean_row = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        mean_row["epoch"] = ep
        mean_row["seconds"] = dt
        history.append(mean_row)
        if ep == 1 or ep % 8 == 0 or ep == epochs:
            print(
                f"  [{arm_name}] ep {ep:2d}/{epochs} ({dt:.2f}s) | loss={mean_row['loss']:.4f} "
                f"l0={mean_row['l0']:.4f} l1={mean_row['l1_base']:.4f} es={mean_row['l_es']:.4f} "
                f"grad={mean_row['grad_norm']:.2f}",
                flush=True,
            )
    total_sec = time.monotonic() - t_start
    print(f"=== FINISHED {arm_name} in {total_sec:.1f}s ({total_sec/60:.2f} min) ===", flush=True)

    # Save EMA checkpoint and world predictions
    ckpt_path = out_dir / "final_ema.pt"
    torch.save(core.cpu_tree(trainer.ema.module.state_dict()), ckpt_path)
    paths, logits = infer_1layer(trainer.ema.module, test_data)
    np.savez_compressed(out_dir / "dev_predictions.npz", paths=paths, logits=logits, ids=test_data["ids"])
    meta = dict(
        arm=arm_name, lambda_es=lambda_es, epochs=epochs, seed=seed,
        seconds=total_sec, n_total=n_total, n_train=n_train,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        history=history,
    )
    (out_dir / "TRAIN_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths, logits, n_total, n_train, total_sec


def verify_against_saved(summary: dict, saved_npz: Path, label: str):
    z = np.load(saved_npz, allow_pickle=True)
    checks = [
        ("minADE20", "minADE", 1.0),
        ("minFDE20", "minFDE", 1.0),
        ("J_cov_pct", "J", 100.0),
        ("Top1_FDE", "Top1_FDE", 1.0),
        ("brier_minFDE", "brier_minFDE", 1.0),
        ("ES_w", "ES", 1.0),
    ]
    for s_k, z_k, scale in checks:
        got = summary[s_k]
        ref = float(z[z_k].mean()) * scale
        diff = abs(got - ref)
        tol = 5e-4 * scale
        if diff > tol:
            raise AssertionError(f"[{label}] verification mismatch on {s_k}: got={got:.6f}, ref={ref:.6f}, diff={diff:.6e}")
    print(f"[VERIFIED] {label} matches {saved_npz.name} within 5e-4", flush=True)


def run_unified_eval(test_data, masks: dict, device: str, seed: int = 42):
    print("\n=== UNIFIED EVALUATION ON OFFICIAL UNIV TEST (N=24,334, seed=42) ===", flush=True)
    results = {}

    # 1. B0_univ_s42
    b0_ck = QROOT / "e_a_diag/results/weights/B0_univ_s42.pt"
    m_b0, _ = core.make_model(device, seed)
    m_b0.load_state_dict(torch.load(b0_ck, map_location="cpu", weights_only=True), strict=True)
    m_b0.requires_grad_(False)
    b0_n = sum(p.numel() for p in m_b0.parameters())
    b0_p, b0_l = infer_b0(m_b0, test_data)
    b0_sum = evaluate_predictions("B0 (Baseline)", b0_p, b0_l, test_data, masks, b0_n, b0_n, b0_motion=None)
    b0_sum["motion_vs_B0_pct"] = {k: 0.0 for k in b0_sum["motion"]}
    verify_against_saved(b0_sum, REV_RUNS / "B0_univ_s42/dev_metrics.npz", "B0_univ_s42")
    results["B0"] = b0_sum
    b0_mot = b0_sum["motion"]
    del m_b0

    # 2. O-conf+E0 & O-conf+QFL (3L saved controls)
    for arm in ("O-conf+E0", "O-conf+QFL"):
        ck = QROOT / "refine_pred/results/weights" / f"{arm}_univ_s42.pt"
        m_q, _ = core.make_model(device, seed)
        attach_qrank(m_q, arm)
        m_q.decoder.use_kin = True
        m_q.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
        m_q.requires_grad_(False)
        q_n = sum(p.numel() for p in m_q.parameters())
        qp, ql = infer_qrank(m_q, test_data)
        qs = evaluate_predictions(f"{arm} (3L)", qp, ql, test_data, masks, q_n, q_n, b0_motion=b0_mot)
        verify_against_saved(qs, REV_RUNS / f"{arm}_univ_s42/dev_metrics.npz", f"{arm}_univ_s42")
        results[arm] = qs
        del m_q

    # 3. Newly trained arms on disk (S2-noES, ES-Calib-1L)
    for arm_key, label in (("S2-noES", "S2-noES (1L)"), ("ES-Calib-1L", "ES-Calib-1L (1L+ES)")):
        pred_file = OUT_ROOT / f"{arm_key}_univ_s{seed}" / "dev_predictions.npz"
        meta_file = OUT_ROOT / f"{arm_key}_univ_s{seed}" / "TRAIN_META.json"
        if not pred_file.exists():
            continue
        pz = np.load(pred_file, allow_pickle=True)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        s = evaluate_predictions(label, pz["paths"], pz["logits"], test_data, masks,
                                 meta["n_total"], meta["n_train"], b0_motion=b0_mot)
        s["train_seconds"] = meta["seconds"]
        results[arm_key] = s

    # Print formatted tables
    print("\n--- TABLE A: Accuracy, Calibration & Proper Scoring Rules (UNIV s42, N=24,334) ---")
    hdr_a = f"{'Model':22s} | {'Params':7s} | {'minADE':6s} {'minFDE':6s} {'J%':5s} | {'Brier':6s} {'B_pen':6s} {'p*':5s} | {'ES_w':6s} {'ES_u':6s} | {'KDE':6s} {'wKDE':6s} | {'hd_wKDE':7s} {'hd_ESw':6s}"
    print(hdr_a)
    print("-" * len(hdr_a))
    for k, s in results.items():
        print(
            f"{s['name']:22s} | {s['total_params']:7d} | {s['minADE20']:6.4f} {s['minFDE20']:6.4f} {s['J_cov_pct']:5.2f} | "
            f"{s['brier_minFDE']:6.4f} {s['B_pen']:6.4f} {s['p_star']:5.3f} | {s['ES_w']:6.4f} {s['ES_u']:6.4f} | "
            f"{s['KDE_NLL']:6.4f} {s['wKDE_NLL']:6.4f} | {s['hard_wKDE_NLL']:7.4f} {s['hard_ES_w']:6.4f}"
        )

    print("\n--- TABLE B: Slot Collapse (Stationary), Ranking & Kinematics vs B0 ---")
    hdr_b = f"{'Model':22s} | {'st_pNr':6s} {'st_#Nr':6s} {'st_H':5s} {'st_mxP':6s} | {'Top1':6s} {'cFDE3':6s} {'rFDE3':6s} | {'spd95%':7s} {'accM%':7s} {'acc95%':7s} {'jrkM%':7s} {'jrk95%':7s}"
    print(hdr_b)
    print("-" * len(hdr_b))
    for k, s in results.items():
        mr = s["motion_vs_B0_pct"]
        print(
            f"{s['name']:22s} | {s['st_p_near_02']:6.3f} {s['st_n_near_02']:6.2f} {s['st_H_p']:5.3f} {s['st_maxP']:6.3f} | "
            f"{s['Top1_FDE']:6.4f} {s['cFDE3']:6.4f} {s['rFDE3']:6.4f} | "
            f"{mr['speed_p95']:+6.2f}% {mr['accel_mean']:+6.2f}% {mr['accel_p95']:+6.2f}% {mr['jerk_mean']:+6.2f}% {mr['jerk_p95']:+6.2f}%"
        )

    out_json = OUT_ROOT / "RESULT_SUMMARY.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved summary -> {out_json}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train", "eval", "auto"), required=True)
    parser.add_argument("--arm", default="ES-Calib-1L")
    parser.add_argument("--lambda-es", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(1)
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda:0"
    print(f"Device: {torch.cuda.get_device_name(0)}, torch={torch.__version__}", flush=True)

    (tr_path, tr_n), (te_path, te_n) = inputs("univ")
    fit_data = load_data(tr_path, tr_n)
    test_data = load_data(te_path, te_n)
    masks = groups(test_data)

    if args.mode == "smoke":
        run_smoke(fit_data, device)
    elif args.mode == "train":
        train_arm(args.arm, args.lambda_es, fit_data, test_data, args.epochs, args.seed, device)
    elif args.mode == "eval":
        run_unified_eval(test_data, masks, device, args.seed)
    elif args.mode == "auto":
        ep_sec = run_smoke(fit_data, device)
        _, _, _, _, sec1 = train_arm("ES-Calib-1L", 0.25, fit_data, test_data, args.epochs, args.seed, device)
        if sec1 <= 900.0:
            train_arm("S2-noES", 0.0, fit_data, test_data, args.epochs, args.seed, device)
        else:
            print(f"[INFO] ES-Calib-1L took {sec1:.1f}s > 900s; skipping S2-noES control for now.", flush=True)
        run_unified_eval(test_data, masks, device, args.seed)


if __name__ == "__main__":
    main()
