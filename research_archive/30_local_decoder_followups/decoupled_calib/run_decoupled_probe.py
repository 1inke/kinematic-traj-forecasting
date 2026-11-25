"""Clean zero-leakage probe: strictly decoupled 1-layer geometry & proper-scoring calibration + adaptive tau(x)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(r"c:\Users\linke\Documents\New project")
WORKSPACE = ROOT / "qoder_work/EXP_QRANK_MLDEC_L4_EXEC_20261122/local/workspace"
QROOT = ROOT / "qoder_work/EXP_QRANK_MLDEC_L4_EXEC_20261122"
REV_RUNS = QROOT / "review/formal_20261123/stage2/runs"
ES_PROBE_OUT = ROOT / "scratch/probe_es_calib_20261124"
OUT_ROOT = ROOT / "scratch/probe_decoupled_calib_20261124"

sys.path.insert(0, str(WORKSPACE))
from advisor_train import core, inputs, load_data  # noqa: E402
from advisor_model import AdvisorDecoder, set_valid, zero  # noqa: E402
from advisor_eval import forecast_metrics, groups  # noqa: E402
from order_numpy import conf_order, risk_order, softmax  # noqa: E402
from qrank_eval import infer as qrank_infer  # noqa: E402
from qrank_model import attach as attach_qrank  # noqa: E402

DT = 0.4


class DecoupledCalibDecoder(AdvisorDecoder):
    """1-layer S2 refinement with hard gradient barrier before proper-scoring & adaptive temperature heads."""

    def __init__(self, h3):
        super().__init__(h3, "S2")
        del self.score
        self.calib_head = nn.Sequential(
            nn.LayerNorm(72),
            nn.Linear(72, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            zero(nn.Linear(64, 1)),
        )
        self.temp_head = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, 32),
            nn.GELU(),
            zero(nn.Linear(32, 1)),
        )
        self.last_dz = None
        self.last_tau = None

    def forward(self, encoded, past):
        b = self.base
        enc, foc = encoded[:, 0], past[:, :1]
        motion_feat = b.motion_features(foc)[:, 0]  # (B, 7)
        c = b.context(torch.cat((enc[:, -1], enc.mean(1), motion_feat), -1))
        q = b.mode_mlp(c[:, None] + b.mode_embedding[None])
        scale = F.softplus(b.proposal_scale(c)) + 0.25
        last = foc[:, 0, -1]
        self.aux, self.order, self.group_weights = None, None, None

        # L0 proposal
        a0 = self.proposal(q, scale)
        paths0 = self.trajectory(a0, last)
        logits0 = b.score_head(torch.cat((q, a0.flatten(2)), -1)).squeeze(-1)
        self.aux = (paths0[:, None], logits0[:, None])

        # L1 geometry refinement
        memory, padding = self.memory(encoded)
        q1 = q + self.geometric((a0 / scale[:, None, None]).flatten(2))
        self.order = torch.argsort(logits0.detach(), dim=1, descending=True, stable=True)
        q1 = q1.gather(1, self.order[..., None].expand_as(q1)) + self.rank[None]
        h = self.layer(q1, memory, padding)
        h = h.gather(1, self.order.argsort(1)[..., None].expand_as(h))
        shift = 0.30 * scale[:, None, None] * torch.tanh(self.offset(h)).reshape(h.size(0), h.size(1), 3, 2)
        a1 = a0 + shift
        paths1 = self.trajectory(a1, last)

        # Strict gradient barrier before calibration & temperature heads
        h_det = h.detach()
        a1_norm_det = (a1 / scale[:, None, None]).detach().flatten(2)
        p1_det = paths1.detach()
        z0_det = logits0.detach()

        ep_det = p1_det[:, :, -1, :]
        dist_ep = torch.cdist(ep_det, ep_det)
        eye = torch.eye(ep_det.size(1), device=ep_det.device, dtype=torch.bool).unsqueeze(0)
        crowd = torch.exp(-0.5 * (dist_ep / 0.25).square()).masked_fill(eye, 0.0).mean(dim=-1, keepdim=True)
        ep_mean = ep_det.mean(dim=1, keepdim=True)
        ep_spread = torch.sqrt(((ep_det - ep_mean).square().sum(dim=-1)).mean(dim=1, keepdim=True) * 0.5 + 1e-6)
        rad_norm = (torch.linalg.vector_norm(ep_det - ep_mean, dim=-1, keepdim=True) / (ep_spread.unsqueeze(-1) + 0.1)).clamp(max=5.0)
        all_mean = p1_det.mean(dim=1, keepdim=True)
        all_spread = torch.sqrt(((p1_det - all_mean).square().sum(dim=-1)).mean(dim=(1, 2)).unsqueeze(-1) * 0.5 + 1e-6)

        calib_in = torch.cat((h_det, a1_norm_det, crowd, rad_norm), dim=-1)
        dz = 1.5 * torch.tanh(self.calib_head(calib_in).squeeze(-1))
        z_raw = z0_det + dz

        z_std = z_raw.detach().std(dim=-1, keepdim=True)
        temp_in = torch.cat((motion_feat.detach(), torch.log(ep_spread + 0.05), torch.log(all_spread + 0.05), z_std), dim=-1)
        raw_lt = self.temp_head(temp_in)
        log_tau = torch.where(raw_lt >= 0.0, 1.4 * torch.tanh(raw_lt / 1.4), 0.7 * torch.tanh(raw_lt / 0.7))
        tau = torch.exp(log_tau)
        z_cal = z_raw / tau

        self.last_dz = dz
        self.last_tau = tau.squeeze(-1)
        return paths1[:, None], z_cal[:, None]


def attach_decoupled(model):
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[]):
        model.decoder = DecoupledCalibDecoder(model.decoder.cpu())
    model.decoder.to(device)
    model.register_forward_pre_hook(set_valid, with_kwargs=True)


def geometry_only_loss(pred: torch.Tensor, target: torch.Tensor, criterion):
    """Exact trajectory-space terms of core.loss_value without classification loss."""
    distances = torch.linalg.vector_norm(pred - target.unsqueeze(-3), dim=-1)
    quality = distances.mean(dim=-1) + criterion.winner_fde_weight * distances[..., -1]
    winner = quality.detach().argmin(dim=-1)
    index = winner[..., None, None, None].expand(*winner.shape, 1, pred.shape[-2], pred.shape[-1])
    winner_prediction = pred.gather(-3, index).squeeze(-3)

    regression = criterion.regression(winner_prediction, target)
    endpoint = F.smooth_l1_loss(winner_prediction[..., -1, :], target[..., -1, :])
    short_steps = min(max(criterion.short_horizon_steps, 1), pred.shape[-2])
    short_horizon = F.smooth_l1_loss(winner_prediction[..., :short_steps, :], target[..., :short_steps, :])

    endpoints = pred[..., -1, :]
    pairwise = torch.cdist(endpoints, endpoints)
    upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
    diversity = F.relu(criterion.diversity_margin - pairwise)[upper].mean() if upper.any() else pred.new_zeros(())

    responsibility = torch.softmax(-quality.detach() / 0.10, dim=-1)
    coverage = (responsibility * quality).sum(dim=-1).mean()

    loss = (
        regression
        + criterion.endpoint_weight * endpoint
        + criterion.short_horizon_weight * short_horizon
        + criterion.diversity_weight * diversity
        + 0.20 * coverage
    )
    detail = dict(
        regression=regression.detach(),
        endpoint=endpoint.detach(),
        short_horizon=short_horizon.detach(),
        diversity=diversity.detach(),
        coverage=coverage.detach(),
    )
    return loss, detail


def gm_nll_loss(pred_det: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Strictly proper step-wise adaptive-bandwidth Gaussian Mixture NLL on detached paths."""
    X = pred_det[:, 0]  # (B, K, T, 2)
    y = target[:, 0].unsqueeze(1)  # (B, 1, T, 2)
    log_p = torch.log_softmax(logits[:, 0], dim=-1)  # (B, K)
    with torch.no_grad():
        mu = X.mean(dim=1, keepdim=True)
        var = ((X - mu) ** 2).sum(dim=-1).mean(dim=1) * 0.5  # (B, T)
        sigma = (X.size(1) ** (-1.0 / 6.0) * torch.sqrt(var + 1e-6)).clamp(min=0.10, max=2.0)  # (B, T)
        dist2 = ((X - y) ** 2).sum(dim=-1)  # (B, K, T)
        log_lik = (
            -math.log(2.0 * math.pi)
            - 2.0 * torch.log(sigma).unsqueeze(1)
            - 0.5 * dist2 / (sigma * sigma).unsqueeze(1)
        )
    log_pdf = torch.logsumexp(log_p.unsqueeze(-1) + log_lik, dim=1)  # (B, T)
    return (-log_pdf).mean()


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


class DecoupledProbeTrainer(core.Trainer):
    """Exact core.Trainer + DecoupledCalibDecoder with gradient isolation."""

    def __init__(self, data, seed: int, device: str, epochs: int):
        super().__init__(data, np.arange(len(data["ids"])), batch=64, epochs=epochs,
                         device=device, seed=seed, order_seed=seed, arm="B0")
        attach_decoupled(self.model)
        self.ema = core.ModelEMA(self.model, 0.995)
        self.optimizer, self.scheduler = core.optimizer_scheduler(self.model, self.steps_per_epoch, epochs)

    def _compute_losses(self, past, tokens, future, augment: bool):
        pred, logits, target, _ = core.prediction(self.model, past, tokens, future, augment=augment)
        aux_p, aux_z = self.model.decoder.aux
        l0, _ = core.loss_value(aux_p, target, aux_z, self.loss_fn)
        l1_geom, g_det = geometry_only_loss(pred, target, self.loss_fn)
        l_gmnll = gm_nll_loss(pred.detach(), logits, target)
        l_es_det = energy_score_loss(pred.detach(), logits, target)
        l_dz_reg = (self.model.decoder.last_dz ** 2).mean()
        l_calib = 0.5 * l_gmnll + 0.5 * l_es_det + 0.01 * l_dz_reg
        loss = 0.5 * l0 + l1_geom + l_calib
        return loss, l0, l1_geom, l_calib, l_gmnll, l_es_det, l_dz_reg, g_det

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
        loss, l0, l1_geom, _, l_gmnll, l_es_det, l_dz_reg, _ = self._compute_losses(past, tokens, future, augment=True)
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
        tau = self.model.decoder.last_tau.detach()
        return dict(
            loss=float(loss.detach()),
            l0=float(l0.detach()),
            l1_geom=float(l1_geom.detach()),
            l_gmnll=float(l_gmnll.detach()),
            l_es_det=float(l_es_det.detach()),
            l_dz_reg=float(l_dz_reg.detach()),
            tau_mean=float(tau.mean()),
            tau_std=float(tau.std(unbiased=False)),
            grad_norm=float(grad),
        )


@torch.no_grad()
def infer_decoupled(model, data, batch: int = 64):
    model.eval()
    device = next(model.parameters()).device
    all_p, all_z, all_ap, all_az, all_tau = [], [], [], [], []
    for lo in range(0, len(data["ids"]), batch):
        sl = slice(lo, lo + batch)
        p, n = (torch.as_tensor(data[k][sl], device=device) for k in ("past", "num_valid"))
        out, z, _, tf = core.prediction(model, p, n)
        aux_p, aux_z = model.decoder.aux
        all_p.append(core.world_paths(out, tf).double().cpu().numpy())
        all_z.append(z[:, 0].double().cpu().numpy())
        all_ap.append(core.world_paths(aux_p, tf).double().cpu().numpy())
        all_az.append(aux_z[:, 0].double().cpu().numpy())
        all_tau.append(model.decoder.last_tau.double().cpu().numpy())
    return (
        np.concatenate(all_p, axis=0),
        np.concatenate(all_z, axis=0),
        np.concatenate(all_ap, axis=0),
        np.concatenate(all_az, axis=0),
        np.concatenate(all_tau, axis=0),
    )


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
    print("=== SMOKE, EXACTNESS & PF-DECOUPLE PRECHECK ===", flush=True)
    smoke_fit = {k: v[:128].copy() for k, v in fit_data.items()}
    trainer = DecoupledProbeTrainer(smoke_fit, seed=42, device=device, epochs=2)
    trainer.model.train()

    past = torch.as_tensor(smoke_fit["past"][:64], device=device)
    future = torch.as_tensor(smoke_fit["future"][:64], device=device)
    tokens = torch.as_tensor(smoke_fit["num_valid"][:64], device=device)

    # 1. Verify energy_score_loss matches NumPy ES
    pred, logits, target, _ = core.prediction(trainer.model, past, tokens, future, augment=False)
    es_torch = float(energy_score_loss(pred, logits, target).detach().cpu())
    es_np = float(compute_es_numpy(pred[:, 0].detach().cpu().numpy(),
                                   logits[:, 0].detach().cpu().numpy(),
                                   target[:, 0].detach().cpu().numpy(),
                                   weighted=True).mean())
    es_diff = abs(es_torch - es_np)
    print(f"[SMOKE-ES] torch={es_torch:.6f}, numpy={es_np:.6f}, abs_diff={es_diff:.3e}", flush=True)
    assert es_diff < 1e-4, f"ES mismatch: {es_diff}"

    # 2. Verify geometry_only_loss matches core.loss_value minus classification
    geom_val, _ = geometry_only_loss(pred, target, trainer.loss_fn)
    full_val, full_det = core.loss_value(pred, target, logits, trainer.loss_fn)
    expected_geom = full_val - trainer.loss_fn.classification_weight * full_det["classification"]
    geom_diff = abs(float(geom_val.detach().cpu()) - float(expected_geom.detach().cpu()))
    print(f"[SMOKE-GEOM] geom={float(geom_val):.6f}, expected={float(expected_geom):.6f}, abs_diff={geom_diff:.3e}", flush=True)
    assert geom_diff < 1e-5, f"geometry_only_loss mismatch: {geom_diff}"

    # Take 2 steps first so zero-initialized output heads (score_head, calib_head, temp_head) are active
    for step_idx in range(2):
        srow = trainer.step()
        print(f"[SMOKE-STEP {step_idx + 1}] loss={srow['loss']:.4f}, tau_mean={srow['tau_mean']:.4f}", flush=True)

    # 3A. PF-DECOUPLE check A: backward ONLY l_calib -> outside calib_head/temp_head must be exact zero
    trainer.model.train()
    _, _, _, l_calib, l_gm, l_esd, _, _ = trainer._compute_losses(past, tokens, future, augment=False)
    trainer.optimizer.zero_grad(set_to_none=True)
    l_calib.backward()
    calib_prefixes = ("decoder.calib_head.", "decoder.temp_head.")
    leak_outside = []
    calib_grad_max = {}
    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        gmax = 0.0 if p.grad is None else float(p.grad.abs().max().item())
        assert gmax == 0.0 or torch.isfinite(p.grad).all(), f"Nonfinite grad on {name}"
        if name.startswith(calib_prefixes):
            head_key = name.split(".")[1]
            calib_grad_max[head_key] = max(calib_grad_max.get(head_key, 0.0), gmax)
        elif gmax != 0.0:
            leak_outside.append((name, gmax))
    print(f"[PF-DECOUPLE-A] l_calib backward: leak_outside={len(leak_outside)}, "
          f"calib_head_max={calib_grad_max.get('calib_head', 0.0):.3e}, "
          f"temp_head_max={calib_grad_max.get('temp_head', 0.0):.3e}", flush=True)
    assert not leak_outside, f"Gradient leaked outside calibration heads: {leak_outside[:5]}"
    assert calib_grad_max.get("calib_head", 0.0) > 0.0, "Zero grad on calib_head!"
    assert calib_grad_max.get("temp_head", 0.0) > 0.0, "Zero grad on temp_head!"

    # 3B. PF-DECOUPLE check B: backward ONLY (0.5*l0 + l1_geom) -> calib_head/temp_head must be exact zero
    _, l0_b, l1_geom_b, _, _, _, _, _ = trainer._compute_losses(past, tokens, future, augment=False)
    trainer.optimizer.zero_grad(set_to_none=True)
    (0.5 * l0_b + l1_geom_b).backward()
    leak_inside = []
    geom_grad_max = {}
    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        gmax = 0.0 if p.grad is None else float(p.grad.abs().max().item())
        assert gmax == 0.0 or torch.isfinite(p.grad).all(), f"Nonfinite grad on {name}"
        if name.startswith(calib_prefixes):
            if gmax != 0.0:
                leak_inside.append((name, gmax))
        elif name.startswith(("decoder.offset.", "decoder.layer.")):
            key = name.split(".")[1]
            geom_grad_max[key] = max(geom_grad_max.get(key, 0.0), gmax)
    print(f"[PF-DECOUPLE-B] geom backward: leak_inside={len(leak_inside)}, "
          f"offset_max={geom_grad_max.get('offset', 0.0):.3e}, "
          f"layer_max={geom_grad_max.get('layer', 0.0):.3e}", flush=True)
    assert not leak_inside, f"Geometry gradient leaked into calib/temp heads: {leak_inside[:5]}"
    assert geom_grad_max.get("offset", 0.0) > 0.0, "Zero grad on decoder.offset!"
    assert geom_grad_max.get("layer", 0.0) > 0.0, "Zero grad on decoder.layer!"

    # 4. 1-epoch univ timing benchmark
    bench_trainer = DecoupledProbeTrainer(fit_data, seed=42, device=device, epochs=64)
    t0 = time.monotonic()
    rows = [bench_trainer.step() for _ in range(bench_trainer.steps_per_epoch)]
    ep_sec = time.monotonic() - t0
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"[TIMING] univ 1 epoch ({bench_trainer.steps_per_epoch} steps) = {ep_sec:.2f}s "
          f"(est 64 epochs = {ep_sec * 64 / 60:.1f} min) | "
          f"loss={m['loss']:.4f} gm={m['l_gmnll']:.4f} es={m['l_es_det']:.4f} tau={m['tau_mean']:.3f}", flush=True)
    for k, v in m.items():
        assert math.isfinite(v), f"Nonfinite epoch stat {k}={v}"
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

    return dict(
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


def train_arm(fit_data, test_data, epochs: int, seed: int, device: str):
    arm_name = "Decoupled-Calib-1L"
    out_dir = OUT_ROOT / f"{arm_name}_univ_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = DecoupledProbeTrainer(fit_data, seed=seed, device=device, epochs=epochs)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n=== TRAIN {arm_name} (epochs={epochs}, seed={seed}, params={n_total}) ===", flush=True)
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
        if ep <= 16 and (mean_row["tau_mean"] < 0.52 or mean_row["tau_mean"] > 3.95):
            raise RuntimeError(
                f"tau(x) pinned near boundary at epoch {ep}: tau_mean={mean_row['tau_mean']:.4f}; stopping for re-plan."
            )
        if ep == 1 or ep % 8 == 0 or ep == epochs:
            print(
                f"  [{arm_name}] ep {ep:2d}/{epochs} ({dt:.2f}s) | loss={mean_row['loss']:.4f} "
                f"l0={mean_row['l0']:.4f} geom={mean_row['l1_geom']:.4f} gm={mean_row['l_gmnll']:.4f} "
                f"es={mean_row['l_es_det']:.4f} dz2={mean_row['l_dz_reg']:.4f} "
                f"tau={mean_row['tau_mean']:.3f}+/-{mean_row['tau_std']:.3f} grad={mean_row['grad_norm']:.2f}",
                flush=True,
            )
    total_sec = time.monotonic() - t_start
    print(f"=== FINISHED {arm_name} in {total_sec:.1f}s ({total_sec/60:.2f} min) ===", flush=True)

    ckpt_path = out_dir / "final_ema.pt"
    torch.save(core.cpu_tree(trainer.ema.module.state_dict()), ckpt_path)
    paths, logits, aux_paths, aux_logits, tau = infer_decoupled(trainer.ema.module, test_data)
    np.savez_compressed(
        out_dir / "dev_predictions.npz",
        paths=paths, logits=logits, aux_paths=aux_paths, aux_logits=aux_logits, tau=tau, ids=test_data["ids"],
    )
    meta = dict(
        arm=arm_name, epochs=epochs, seed=seed,
        seconds=total_sec, n_total=n_total, n_train=n_train,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        history=history,
    )
    (out_dir / "TRAIN_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def eval_es_only(paths: np.ndarray, logits: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None):
    if mask is not None:
        P, Z, g = paths[mask], logits[mask], gt[mask]
    else:
        P, Z, g = paths, logits, gt
    es = compute_es_numpy(P, Z, g, weighted=True)
    w = softmax(Z)
    fde = np.linalg.norm(P[:, :, -1] - g[:, None, -1], axis=-1)
    top1 = fde[np.arange(len(P)), w.argmax(axis=1)]
    best_i = fde.argmin(axis=1)
    pstar = w[np.arange(len(P)), best_i]
    brier = fde[np.arange(len(P)), best_i] + (1.0 - pstar) ** 2
    return dict(
        ES_w=float(es.mean()),
        Top1_FDE=float(top1.mean()),
        minFDE20=float(fde.min(axis=1).mean()),
        Brier=float(brier.mean()),
    )


def run_unified_eval(fit_data, test_data, masks: dict, device: str, seed: int = 42):
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

    # 3. ES-Calib-1L (Coupled 1L saved predictions)
    es_pred = ES_PROBE_OUT / f"ES-Calib-1L_univ_s{seed}/dev_predictions.npz"
    es_meta = ES_PROBE_OUT / f"ES-Calib-1L_univ_s{seed}/TRAIN_META.json"
    if es_pred.exists():
        epz = np.load(es_pred, allow_pickle=True)
        em = json.loads(es_meta.read_text(encoding="utf-8"))
        es_s = evaluate_predictions(
            "ES-Calib-1L (Coupl)", epz["paths"], epz["logits"], test_data, masks,
            em["n_total"], em["n_train"], b0_motion=b0_mot,
        )
        es_s["train_seconds"] = em["seconds"]
        results["ES-Calib-1L"] = es_s

    # 4. Decoupled-Calib-1L (Ours L1 & L0 aux)
    dec_dir = OUT_ROOT / f"Decoupled-Calib-1L_univ_s{seed}"
    dec_pred = np.load(dec_dir / "dev_predictions.npz", allow_pickle=True)
    dec_meta = json.loads((dec_dir / "TRAIN_META.json").read_text(encoding="utf-8"))
    dec_s = evaluate_predictions(
        "Decoupled-Calib (L1)", dec_pred["paths"], dec_pred["logits"], test_data, masks,
        dec_meta["n_total"], dec_meta["n_train"], b0_motion=b0_mot,
    )
    dec_s["train_seconds"] = dec_meta["seconds"]
    results["Decoupled-Calib-1L"] = dec_s

    dec_l0_s = evaluate_predictions(
        "Decoupled-Calib (L0)", dec_pred["aux_paths"], dec_pred["aux_logits"], test_data, masks,
        dec_meta["n_total"], dec_meta["n_train"], b0_motion=b0_mot,
    )
    results["Decoupled-Calib-L0"] = dec_l0_s

    print("\n--- TABLE A: Accuracy, Calibration & Proper Scoring Rules (UNIV s42, N=24,334) ---")
    hdr_a = f"{'Model':22s} | {'Params':7s} | {'minADE':6s} {'minFDE':6s} {'J%':5s} | {'Brier':6s} {'B_pen':6s} {'p*':5s} | {'ES_w':6s} {'ES_u':6s} | {'KDE':6s} {'wKDE':6s} | {'hd_wKDE':7s} {'hd_ESw':6s}"
    print(hdr_a)
    print("-" * len(hdr_a))
    for _, s in results.items():
        print(
            f"{s['name']:22s} | {s['total_params']:7d} | {s['minADE20']:6.4f} {s['minFDE20']:6.4f} {s['J_cov_pct']:5.2f} | "
            f"{s['brier_minFDE']:6.4f} {s['B_pen']:6.4f} {s['p_star']:5.3f} | {s['ES_w']:6.4f} {s['ES_u']:6.4f} | "
            f"{s['KDE_NLL']:6.4f} {s['wKDE_NLL']:6.4f} | {s['hard_wKDE_NLL']:7.4f} {s['hard_ES_w']:6.4f}"
        )

    print("\n--- TABLE B: Slot Collapse (Stationary), Ranking & Kinematics vs B0 ---")
    hdr_b = f"{'Model':22s} | {'st_pNr':6s} {'st_#Nr':6s} {'st_H':5s} {'st_mxP':6s} | {'Top1':6s} {'cFDE3':6s} {'rFDE3':6s} | {'rFDE5':6s} | {'spd95%':7s} {'accM%':7s} {'jrk95%':7s} {'jAcc95%':7s}"
    print(hdr_b)
    print("-" * len(hdr_b))
    for _, s in results.items():
        mr = s["motion_vs_B0_pct"]
        print(
            f"{s['name']:22s} | {s['st_p_near_02']:6.3f} {s['st_n_near_02']:6.2f} {s['st_H_p']:5.3f} {s['st_maxP']:6.3f} | "
            f"{s['Top1_FDE']:6.4f} {s['cFDE3']:6.4f} {s['rFDE3']:6.4f} | {s['rFDE5']:6.4f} | "
            f"{mr['speed_p95']:+6.2f}% {mr['accel_mean']:+6.2f}% {mr['jerk_p95']:+6.2f}% {mr['junction_accel_p95']:+6.2f}%"
        )

    # Table C: tau(x) diagnostics across subsets
    tau = dec_pred["tau"].astype(np.float64)
    st_m = masks["stationary"].astype(bool)
    sd_m = masks["steady"].astype(bool)
    mv_m = ~st_m
    hd_m = masks["hard"].astype(bool)
    tau_diag = {}
    print("\n--- TABLE C: Learned State-Adaptive Temperature tau(x) on UNIV Test ---")
    print(f"{'Subset':12s} | {'N':6s} | {'mean':6s} {'median':6s} {'p10':6s} {'p90':6s}")
    print("-" * 52)
    for sub, msk in [("overall", np.ones(len(tau), bool)), ("stationary", st_m), ("steady", sd_m), ("moving", mv_m), ("hard", hd_m)]:
        t_sub = tau[msk]
        row = dict(
            N=int(msk.sum()),
            mean=float(t_sub.mean()),
            median=float(np.median(t_sub)),
            p10=float(np.quantile(t_sub, 0.10)),
            p90=float(np.quantile(t_sub, 0.90)),
        )
        tau_diag[sub] = row
        print(f"{sub:12s} | {row['N']:6d} | {row['mean']:6.4f} {row['median']:6.4f} {row['p10']:6.4f} {row['p90']:6.4f}")

    # Table D: Train-Sub (N=4000) vs Test-Univ (N=24,334) Overfitting Check
    m_dec, _ = core.make_model(device, seed)
    attach_decoupled(m_dec)
    m_dec.load_state_dict(torch.load(dec_dir / "final_ema.pt", map_location="cpu", weights_only=True), strict=True)
    m_dec.requires_grad_(False)
    tr_sub = {k: v[:4000] for k, v in fit_data.items()}
    tr_p1, tr_z1, tr_p0, tr_z0, _ = infer_decoupled(m_dec, tr_sub)
    del m_dec
    tr_masks = groups(tr_sub)
    tr_gt = tr_sub["future"].astype(np.float64)
    te_gt = test_data["future"].astype(np.float64)
    te_p1, te_z1 = dec_pred["paths"], dec_pred["logits"]
    te_p0, te_z0 = dec_pred["aux_paths"], dec_pred["aux_logits"]

    overfit_diag = {}
    print("\n--- TABLE D: Train-Sub (N=4000) vs Test-Univ (N=24,334) Generalization Check ---")
    for split_name, p0, z0, p1, z1, gt_s, m_s in [
        ("TRAIN_SUB", tr_p0, tr_z0, tr_p1, tr_z1, tr_gt, tr_masks),
        ("TEST_UNIV", te_p0, te_z0, te_p1, te_z1, te_gt, masks),
    ]:
        print(f"=== {split_name} ===")
        split_dict = {}
        st_s = m_s["stationary"].astype(bool)
        hd_s = m_s["hard"].astype(bool)
        for sub_name, msk in [("ALL", None), ("STAT", st_s), ("MOVE", ~st_s), ("HARD", hd_s)]:
            e0 = eval_es_only(p0, z0, gt_s, msk)
            e1 = eval_es_only(p1, z1, gt_s, msk)
            e10 = eval_es_only(p1, z0, gt_s, msk)
            split_dict[sub_name] = dict(L0=e0, L1=e1, L1_plus_z0=e10)
            print(
                f"  {sub_name:4s} | L0: ES={e0['ES_w']:.4f} Top1={e0['Top1_FDE']:.4f} mFDE={e0['minFDE20']:.4f} Br={e0['Brier']:.4f} | "
                f"L1: ES={e1['ES_w']:.4f} Top1={e1['Top1_FDE']:.4f} mFDE={e1['minFDE20']:.4f} Br={e1['Brier']:.4f} | "
                f"L1+z0: ES={e10['ES_w']:.4f} Top1={e10['Top1_FDE']:.4f}"
            )
        overfit_diag[split_name] = split_dict

    out_payload = dict(results=results, tau_diagnostics=tau_diag, train_vs_test=overfit_diag)
    out_json = OUT_ROOT / "RESULT_SUMMARY.json"
    out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"\nSaved summary -> {out_json}", flush=True)
    return out_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train", "eval", "train_eval"), required=True)
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
        train_arm(fit_data, test_data, args.epochs, args.seed, device)
    elif args.mode == "eval":
        run_unified_eval(fit_data, test_data, masks, device, args.seed)
    elif args.mode == "train_eval":
        train_arm(fit_data, test_data, args.epochs, args.seed, device)
        run_unified_eval(fit_data, test_data, masks, device, args.seed)


if __name__ == "__main__":
    main()
