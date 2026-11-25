"""Clean zero-leakage probe: Version 1 MCUS-Decoupled-1L across univ + budget-checked eth/C0.

Original decoder mechanism:
  1. Endpoint Marginal Coverage Utility Reordering (greedy facility-location risk_order)
  2. Endpoint Spatial-Overlap Biased Causal Self-Attention (2-scale Gaussian overlap bias)
  3. Redundancy + Motion-Gated Residual Refinement (prevents high-speed over-refinement & slot collapse)
  4. Strictly Decoupled Proper-Scoring Calibration (fixed tau=1.0 during training; GM-NLL + Energy Score)
"""
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
C0_RUNS = ROOT / "qoder_work/EXP_ADVISOR_3DIR_L4_EXEC_20261121/local/results/stage1/runs"
ES_PROBE_OUT = ROOT / "scratch/probe_es_calib_20261124"
DEC_PROBE_OUT = ROOT / "scratch/probe_decoupled_calib_20261124"
OUT_ROOT = ROOT / "scratch/probe_mcus_20261124"

sys.path.insert(0, str(WORKSPACE))
from advisor_train import core, inputs, load_data  # noqa: E402
from advisor_model import AdvisorDecoder, set_valid, zero  # noqa: E402
from advisor_eval import forecast_metrics, groups  # noqa: E402
from order_numpy import conf_order, risk_order as np_risk_order, softmax  # noqa: E402
from qrank_eval import infer as qrank_infer  # noqa: E402
from qrank_model import attach as attach_qrank, risk_order as qrank_risk_order  # noqa: E402

DT = 0.4


@torch.no_grad()
def risk_order_fast(end: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact greedy facility-location marginal risk-reduction ordering in FP32."""
    end = end.float()
    w = w.float()
    B, K, _ = end.shape
    rows = torch.arange(B, device=end.device)
    D = torch.cdist(end, end)
    best = torch.full((B, K), float("inf"), dtype=D.dtype, device=D.device)
    taken = torch.zeros(B, K, dtype=torch.bool, device=D.device)
    tie = torch.arange(K, device=D.device, dtype=D.dtype) * 1e-6
    out = []
    for _ in range(K):
        obj = (w[:, None, :] * torch.minimum(best[:, None, :], D)).sum(-1) + tie
        obj = obj.masked_fill(taken, float("inf"))
        c = obj.argmin(1)
        out.append(c)
        taken[rows, c] = True
        best = torch.minimum(best, D[rows, c])
    return torch.stack(out, 1)


class MCUSCrossLayer(nn.Module):
    """Causal CrossLayer with per-head 2-scale endpoint spatial-overlap attention bias."""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.self_norm = nn.LayerNorm(64)
        self.overlap_bias = nn.Linear(2, 4, bias=False)
        nn.init.zeros_(self.overlap_bias.weight)
        self.cross_norm = nn.LayerNorm(64)
        self.cross = nn.MultiheadAttention(64, 4, batch_first=True)
        self.ff_norm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, q: torch.Tensor, memory: torch.Tensor, padding: torch.Tensor, dist_ord: torch.Tensor):
        B, K, _ = q.shape
        h = self.self_norm(q)
        eye = torch.eye(K, device=q.device, dtype=torch.bool).unsqueeze(0)
        phi1 = torch.exp(-0.5 * (dist_ord / 0.35).square()).masked_fill(eye, 0.0)
        phi2 = torch.exp(-0.5 * (dist_ord / 1.00).square()).masked_fill(eye, 0.0)
        bias = self.overlap_bias(torch.stack((phi1, phi2), dim=-1)).permute(0, 3, 1, 2)  # (B, 4, K, K)
        upper = torch.ones(K, K, device=q.device, dtype=torch.bool).triu(1)
        attn_mask = bias.masked_fill(upper[None, None], float("-inf")).reshape(B * 4, K, K)
        q = q + self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        q = q + self.cross(self.cross_norm(q), memory, memory, key_padding_mask=padding, need_weights=False)[0]
        return q + self.ff(self.ff_norm(q))


class MCUSDecoupledDecoder(AdvisorDecoder):
    """MCUS reordering + overlap-biased causal self-attn + redundancy/motion gate + decoupled calib."""

    def __init__(self, h3):
        super().__init__(h3, "S2")
        del self.score
        self.layer = MCUSCrossLayer()
        self.ref_gate = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.ref_gate[-1].weight)
        nn.init.constant_(self.ref_gate[-1].bias, 1.0)
        self.calib_head = nn.Sequential(
            nn.LayerNorm(72),
            nn.Linear(72, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            zero(nn.Linear(64, 1)),
        )
        self.last_dz = None
        self.last_gate = None
        self.last_ov = None

    def forward(self, encoded: torch.Tensor, past: torch.Tensor):
        b = self.base
        enc, foc = encoded[:, 0], past[:, :1]
        motion_feat = b.motion_features(foc)[:, 0]  # (B, 7)
        c = b.context(torch.cat((enc[:, -1], enc.mean(1), motion_feat), -1))
        q = b.mode_mlp(c[:, None] + b.mode_embedding[None])
        scale = F.softplus(b.proposal_scale(c)) + 0.25
        last = foc[:, 0, -1]
        self.aux, self.order, self.group_weights = None, None, None

        # 1. L0 proposal
        a0 = self.proposal(q, scale)
        paths0 = self.trajectory(a0, last)
        logits0 = b.score_head(torch.cat((q, a0.flatten(2)), -1)).squeeze(-1)
        self.aux = (paths0[:, None], logits0[:, None])

        # 2. Marginal Coverage Utility Reordering + predecessor redundancy features
        with torch.no_grad():
            ep0 = paths0[:, :, -1, :].detach()
            p0 = torch.softmax(logits0.detach().float(), dim=-1)
            order = risk_order_fast(ep0, p0)
            self.order = order
            ep_ord = ep0.gather(1, order[..., None].expand_as(ep0))
            dist_ord = torch.cdist(ep_ord, ep_ord)
            triu0 = torch.ones(ep_ord.size(1), ep_ord.size(1), device=ep_ord.device, dtype=torch.bool).triu(0)
            d_prev_ord = dist_ord.masked_fill(triu0.unsqueeze(0), float("inf")).amin(dim=-1)
            d_prev_ord[:, 0] = 2.0
            ov_ord = torch.exp(-0.5 * (d_prev_ord / 0.35).square())
            ov_ord[:, 0] = 0.0

        # 3. L1 Overlap-Biased Causal Refinement + Redundancy/Motion Gate
        memory, padding = self.memory(encoded)
        q1 = (q + self.geometric((a0 / scale[:, None, None]).flatten(2))).gather(
            1, order[..., None].expand_as(q)
        ) + self.rank[None]
        h_ord = self.layer(q1, memory, padding, dist_ord)
        inverse = order.argsort(1)
        h = h_ord.gather(1, inverse[..., None].expand_as(h_ord))

        ov_orig = ov_ord.gather(1, inverse).unsqueeze(-1)
        d_prev_orig = d_prev_ord.gather(1, inverse).unsqueeze(-1)
        rank_norm = (inverse.to(q.dtype) / max(q.size(1) - 1, 1)).unsqueeze(-1)
        mot_exp = motion_feat.detach().unsqueeze(1).expand(-1, q.size(1), -1)
        gate_in = torch.cat((mot_exp, ov_orig, rank_norm, torch.log(d_prev_orig + 0.05)), dim=-1)
        gate = torch.sigmoid(self.ref_gate(gate_in))  # (B, K, 1)

        shift = 0.35 * gate.unsqueeze(-1) * scale[:, None, None] * torch.tanh(self.offset(h)).reshape(
            h.size(0), h.size(1), 3, 2
        )
        a1 = a0 + shift
        paths1 = self.trajectory(a1, last)

        # 4. Strict gradient barrier + decoupled proper-scoring calibration head (tau=1.0)
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

        calib_in = torch.cat((h_det, a1_norm_det, crowd, rad_norm), dim=-1)
        dz = 1.5 * torch.tanh(self.calib_head(calib_in).squeeze(-1))
        z_cal = z0_det + dz

        self.last_dz = dz
        self.last_gate = gate.squeeze(-1).detach()
        self.last_ov = ov_orig.squeeze(-1).detach()
        return paths1[:, None], z_cal[:, None]


def attach_mcus(model):
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[]):
        model.decoder = MCUSDecoupledDecoder(model.decoder.cpu())
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
    detail = dict(regression=regression.detach())
    return loss, detail


def gm_nll_loss(pred_det: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Strictly proper step-wise adaptive-bandwidth Gaussian Mixture NLL on detached paths."""
    X = pred_det[:, 0]
    y = target[:, 0].unsqueeze(1)
    log_p = torch.log_softmax(logits[:, 0], dim=-1)
    with torch.no_grad():
        mu = X.mean(dim=1, keepdim=True)
        var = ((X - mu) ** 2).sum(dim=-1).mean(dim=1) * 0.5
        sigma = (X.size(1) ** (-1.0 / 6.0) * torch.sqrt(var + 1e-6)).clamp(min=0.10, max=2.0)
        dist2 = ((X - y) ** 2).sum(dim=-1)
        log_lik = (
            -math.log(2.0 * math.pi)
            - 2.0 * torch.log(sigma).unsqueeze(1)
            - 0.5 * dist2 / (sigma * sigma).unsqueeze(1)
        )
    log_pdf = torch.logsumexp(log_p.unsqueeze(-1) + log_lik, dim=1)
    return (-log_pdf).mean()


def energy_score_loss(pred: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Differentiable 24D Energy Score loss: E_p||X - y|| - 0.5 E_{p,p}||X - X'||."""
    X = pred[:, 0].reshape(pred.size(0), pred.size(2), -1)
    y = target[:, 0].unsqueeze(1).reshape(target.size(0), 1, -1)
    p = torch.softmax(logits[:, 0], dim=-1)
    d_xy = torch.sqrt(((X - y) ** 2).sum(dim=-1) + eps * eps)
    term1 = (p * d_xy).sum(dim=-1)
    diff = X.unsqueeze(2) - X.unsqueeze(1)
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
    """All-K20 pooled kinematics (dt=0.4s, prepending last observed point)."""
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


class MCUSProbeTrainer(core.Trainer):
    """Exact core.Trainer + MCUSDecoupledDecoder with GPU-resident batches & foreach=True."""

    def __init__(self, data, seed: int, device: str, epochs: int):
        super().__init__(data, np.arange(len(data["ids"])), batch=64, epochs=epochs,
                         device=device, seed=seed, order_seed=seed, arm="B0")
        attach_mcus(self.model)
        self.ema = core.ModelEMA(self.model, 0.995)
        self.optimizer, self.scheduler = core.optimizer_scheduler(self.model, self.steps_per_epoch, epochs)
        for g in self.optimizer.param_groups:
            g["foreach"] = True
        self.optimizer.defaults["foreach"] = True
        self.past_gpu = torch.as_tensor(data["past"], device=device)
        self.future_gpu = torch.as_tensor(data["future"], device=device)
        self.tokens_gpu = torch.as_tensor(data["num_valid"], device=device)

    def _compute_losses(self, past, tokens, future, augment: bool):
        pred, logits, target, _ = core.prediction(self.model, past, tokens, future, augment=augment)
        aux_p, aux_z = self.model.decoder.aux
        l0, _ = core.loss_value(aux_p, target, aux_z, self.loss_fn)
        l1_geom, _ = geometry_only_loss(pred, target, self.loss_fn)
        l_gmnll = gm_nll_loss(pred.detach(), logits, target)
        l_es_det = energy_score_loss(pred.detach(), logits, target)
        l_dz_reg = (self.model.decoder.last_dz ** 2).mean()
        l_calib = 0.5 * l_gmnll + 0.5 * l_es_det + 0.01 * l_dz_reg
        loss = 0.5 * l0 + l1_geom + l_calib
        return loss, l0, l1_geom, l_calib, l_gmnll, l_es_det, l_dz_reg

    def step(self):
        if self.order is None or self.cursor == len(self.order):
            if self.epoch >= self.epochs:
                raise StopIteration
            self.order = self.indices[torch.randperm(len(self.indices), generator=self.generator).numpy()]
            self.cursor, self.epoch = 0, self.epoch + 1
        idx = self.order[self.cursor:self.cursor + self.batch]
        self.model.train()
        past, future, tokens = self.past_gpu[idx], self.future_gpu[idx], self.tokens_gpu[idx]
        loss, l0, l1_geom, _, l_gmnll, l_es_det, l_dz_reg = self._compute_losses(past, tokens, future, augment=True)
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
        gate = self.model.decoder.last_gate
        ov = self.model.decoder.last_ov
        return dict(
            loss=float(loss.detach()),
            l0=float(l0.detach()),
            l1_geom=float(l1_geom.detach()),
            l_gmnll=float(l_gmnll.detach()),
            l_es_det=float(l_es_det.detach()),
            l_dz_reg=float(l_dz_reg.detach()),
            gate_mean=float(gate.mean()),
            gate_std=float(gate.std(unbiased=False)),
            ov_mean=float(ov.mean()),
            grad_norm=float(grad),
        )


@torch.no_grad()
def infer_mcus(model, data, batch: int = 64):
    model.eval()
    device = next(model.parameters()).device
    all_p, all_z, all_ap, all_az, all_g, all_ov = [], [], [], [], [], []
    for lo in range(0, len(data["ids"]), batch):
        sl = slice(lo, lo + batch)
        p, n = (torch.as_tensor(data[k][sl], device=device) for k in ("past", "num_valid"))
        out, z, _, tf = core.prediction(model, p, n)
        aux_p, aux_z = model.decoder.aux
        all_p.append(core.world_paths(out, tf).double().cpu().numpy())
        all_z.append(z[:, 0].double().cpu().numpy())
        all_ap.append(core.world_paths(aux_p, tf).double().cpu().numpy())
        all_az.append(aux_z[:, 0].double().cpu().numpy())
        all_g.append(model.decoder.last_gate.double().cpu().numpy())
        all_ov.append(model.decoder.last_ov.double().cpu().numpy())
    return (
        np.concatenate(all_p, axis=0),
        np.concatenate(all_z, axis=0),
        np.concatenate(all_ap, axis=0),
        np.concatenate(all_az, axis=0),
        np.concatenate(all_g, axis=0),
        np.concatenate(all_ov, axis=0),
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
    print("=== SMOKE, EXACTNESS, PF-DECOUPLE & MECHANISM CHECK ===", flush=True)
    smoke_fit = {k: v[:128].copy() for k, v in fit_data.items()}
    trainer = MCUSProbeTrainer(smoke_fit, seed=42, device=device, epochs=2)
    trainer.model.train()

    past = torch.as_tensor(smoke_fit["past"][:64], device=device)
    future = torch.as_tensor(smoke_fit["future"][:64], device=device)
    tokens = torch.as_tensor(smoke_fit["num_valid"][:64], device=device)

    # 1. Verify energy_score_loss & geometry_only_loss exactness
    pred, logits, target, _ = core.prediction(trainer.model, past, tokens, future, augment=False)
    es_torch = float(energy_score_loss(pred, logits, target).detach().cpu())
    es_np = float(compute_es_numpy(pred[:, 0].detach().cpu().numpy(),
                                   logits[:, 0].detach().cpu().numpy(),
                                   target[:, 0].detach().cpu().numpy(),
                                   weighted=True).mean())
    es_diff = abs(es_torch - es_np)
    print(f"[SMOKE-ES] torch={es_torch:.6f}, numpy={es_np:.6f}, abs_diff={es_diff:.3e}", flush=True)
    assert es_diff < 1e-4, f"ES mismatch: {es_diff}"

    geom_val, _ = geometry_only_loss(pred, target, trainer.loss_fn)
    full_val, full_det = core.loss_value(pred, target, logits, trainer.loss_fn)
    expected_geom = full_val - trainer.loss_fn.classification_weight * full_det["classification"]
    geom_diff = abs(float(geom_val.detach().cpu()) - float(expected_geom.detach().cpu()))
    print(f"[SMOKE-GEOM] geom={float(geom_val):.6f}, expected={float(expected_geom):.6f}, abs_diff={geom_diff:.3e}", flush=True)
    assert geom_diff < 1e-5, f"geometry_only_loss mismatch: {geom_diff}"

    # 2. Take 2 warmup steps & verify risk_order_fast matches qrank_model.risk_order >= 98%
    for step_idx in range(2):
        srow = trainer.step()
        assert math.isfinite(srow["loss"]) and math.isfinite(srow["gate_mean"])
        print(f"[SMOKE-STEP {step_idx + 1}] loss={srow['loss']:.4f}, gate_mean={srow['gate_mean']:.4f}", flush=True)

    with torch.no_grad():
        ep_test = pred[:, 0, :, -1, :].detach()
        w_test = torch.softmax(torch.randn(ep_test.size(0), ep_test.size(1), device=device), dim=-1)
        ord_fast = risk_order_fast(ep_test, w_test)
        ord_ref = qrank_risk_order(ep_test, w_test)
        match_rate = float((ord_fast == ord_ref).float().mean().item())
        print(f"[SMOKE-RISK-ORDER] fast FP32 vs FP64 exact entry match={match_rate * 100:.2f}%", flush=True)
        assert match_rate >= 0.98, f"risk_order_fast match rate too low: {match_rate}"

    # 3A. PF-DECOUPLE check A: backward ONLY l_calib -> outside calib_head must be exact zero
    trainer.model.train()
    _, _, _, l_calib, _, _, _ = trainer._compute_losses(past, tokens, future, augment=False)
    trainer.optimizer.zero_grad(set_to_none=True)
    l_calib.backward()
    leak_outside = []
    calib_max = 0.0
    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        gmax = 0.0 if p.grad is None else float(p.grad.abs().max().item())
        assert gmax == 0.0 or torch.isfinite(p.grad).all(), f"Nonfinite grad on {name}"
        if name.startswith("decoder.calib_head."):
            calib_max = max(calib_max, gmax)
        elif gmax != 0.0:
            leak_outside.append((name, gmax))
    print(f"[PF-DECOUPLE-A] l_calib backward: leak_outside={len(leak_outside)}, calib_head_max={calib_max:.3e}", flush=True)
    assert not leak_outside, f"Gradient leaked outside calib_head: {leak_outside[:5]}"
    assert calib_max > 0.0, "Zero grad on calib_head!"

    # 3B. PF-DECOUPLE check B: backward ONLY (0.5*l0 + l1_geom) -> calib_head zero; mechanisms > 0
    _, l0_b, l1_geom_b, _, _, _, _ = trainer._compute_losses(past, tokens, future, augment=False)
    trainer.optimizer.zero_grad(set_to_none=True)
    (0.5 * l0_b + l1_geom_b).backward()
    leak_inside = []
    req_grads = {"decoder.offset": 0.0, "decoder.layer": 0.0, "decoder.layer.overlap_bias": 0.0, "decoder.ref_gate": 0.0}
    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        gmax = 0.0 if p.grad is None else float(p.grad.abs().max().item())
        assert gmax == 0.0 or torch.isfinite(p.grad).all(), f"Nonfinite grad on {name}"
        if name.startswith("decoder.calib_head."):
            if gmax != 0.0:
                leak_inside.append((name, gmax))
        for pref in req_grads:
            if name.startswith(pref):
                req_grads[pref] = max(req_grads[pref], gmax)
    print(f"[PF-DECOUPLE-B] geom backward: leak_inside={len(leak_inside)}, grads={ {k: f'{v:.2e}' for k, v in req_grads.items()} }", flush=True)
    assert not leak_inside, f"Geometry gradient leaked into calib_head: {leak_inside[:5]}"
    for pref, gmax in req_grads.items():
        assert gmax > 0.0, f"Zero grad on {pref}!"

    # 4. 1-epoch univ GPU timing benchmark
    bench_trainer = MCUSProbeTrainer(fit_data, seed=42, device=device, epochs=64)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    rows = [bench_trainer.step() for _ in range(bench_trainer.steps_per_epoch)]
    torch.cuda.synchronize()
    ep_sec = time.monotonic() - t0
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    univ_est_min = ep_sec * 64 / 60.0
    eth_est_min = ep_sec * (474.0 / 155.0) * 64 / 60.0
    print(
        f"[TIMING] univ 1 epoch ({bench_trainer.steps_per_epoch} steps) = {ep_sec:.2f}s "
        f"(est univ 64ep={univ_est_min:.1f}m, eth 64ep={eth_est_min:.1f}m, sum={univ_est_min + eth_est_min:.1f}m) | "
        f"loss={m['loss']:.4f} gm={m['l_gmnll']:.4f} es={m['l_es_det']:.4f} gate={m['gate_mean']:.3f}",
        flush=True,
    )
    for k, v in m.items():
        assert math.isfinite(v), f"Nonfinite epoch stat {k}={v}"
    if ep_sec > 28.0:
        raise RuntimeError(f"1-epoch time {ep_sec:.2f}s > 28s budget; stopping for re-plan.")
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
    r_ord = np_risk_order(ep, p)
    c_fde = np.take_along_axis(fde, c_ord, axis=1)
    r_fde = np.take_along_axis(fde, r_ord, axis=1)

    st = masks["stationary"].astype(bool)
    hd = masks["hard"].astype(bool)

    mot = motion_stats(paths, data["past"][:, 0].astype(np.float64))
    mot_rel = {}
    if b0_motion is not None:
        for k, v in mot.items():
            mot_rel[k] = (v - b0_motion[k]) / b0_motion[k] * 100.0

    def m_opt(arr, msk):
        return float(arr[msk].mean()) if msk.any() else None

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
        st_p_near_02=m_opt(p_near, st),
        st_n_near_02=m_opt(n_near, st),
        st_p_star=m_opt(p_star, st),
        st_H_p=m_opt(ent, st),
        st_maxP=m_opt(maxp, st),
        ES_w=float(es_w.mean()),
        ES_u=float(es_u.mean()),
        KDE_NLL=float(kde.mean()),
        wKDE_NLL=float(wkde.mean()),
        hard_ES_w=m_opt(es_w, hd),
        hard_ES_u=m_opt(es_u, hd),
        hard_KDE_NLL=m_opt(kde, hd),
        hard_wKDE_NLL=m_opt(wkde, hd),
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


def train_fold(fold: str, fit_data, test_data, epochs: int, seed: int, device: str):
    arm_name = "MCUS-Decoupled-1L"
    out_dir = OUT_ROOT / f"{arm_name}_{fold}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = MCUSProbeTrainer(fit_data, seed=seed, device=device, epochs=epochs)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n=== TRAIN {arm_name} [{fold}] (epochs={epochs}, steps/ep={trainer.steps_per_epoch}, seed={seed}, params={n_total}) ===", flush=True)
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
                f"  [{fold}] ep {ep:2d}/{epochs} ({dt:.2f}s) | loss={mean_row['loss']:.4f} "
                f"l0={mean_row['l0']:.4f} geom={mean_row['l1_geom']:.4f} gm={mean_row['l_gmnll']:.4f} "
                f"es={mean_row['l_es_det']:.4f} gate={mean_row['gate_mean']:.3f}+/-{mean_row['gate_std']:.3f} grad={mean_row['grad_norm']:.2f}",
                flush=True,
            )
    total_sec = time.monotonic() - t_start
    print(f"=== FINISHED {arm_name} [{fold}] in {total_sec:.1f}s ({total_sec / 60:.2f} min) ===", flush=True)

    ckpt_path = out_dir / "final_ema.pt"
    torch.save(core.cpu_tree(trainer.ema.module.state_dict()), ckpt_path)
    paths, logits, aux_paths, aux_logits, gate, ov = infer_mcus(trainer.ema.module, test_data)
    np.savez_compressed(
        out_dir / "dev_predictions.npz",
        paths=paths, logits=logits, aux_paths=aux_paths, aux_logits=aux_logits, gate=gate, ov=ov, ids=test_data["ids"],
    )
    ob_w = trainer.ema.module.decoder.layer.overlap_bias.weight.detach().cpu().tolist()
    meta = dict(
        arm=arm_name, fold=fold, epochs=epochs, steps_per_epoch=trainer.steps_per_epoch, seed=seed,
        seconds=total_sec, n_total=n_total, n_train=n_train,
        overlap_bias_weight=ob_w,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        history=history,
    )
    (out_dir / "TRAIN_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def fmt_opt(val, spec):
    return format(val, spec) if val is not None else "   N/A"


def print_tables(fold_label: str, results: dict, mc_pred: dict, mc_meta: dict, masks: dict):
    print(f"\n--- TABLE A ({fold_label}): Accuracy, Calibration & Proper Scoring Rules ---")
    hdr_a = f"{'Model':26s} | {'Params':7s} | {'minADE':6s} {'minFDE':6s} {'J%':5s} | {'Brier':6s} {'B_pen':6s} {'p*':5s} | {'ES_w':6s} {'ES_u':6s} | {'KDE':6s} {'wKDE':6s} | {'hd_wKDE':7s} {'hd_ESw':6s}"
    print(hdr_a)
    print("-" * len(hdr_a))
    for s in results.values():
        print(
            f"{s['name']:26s} | {s['total_params']:7d} | {s['minADE20']:6.4f} {s['minFDE20']:6.4f} {s['J_cov_pct']:5.2f} | "
            f"{s['brier_minFDE']:6.4f} {s['B_pen']:6.4f} {s['p_star']:5.3f} | {s['ES_w']:6.4f} {s['ES_u']:6.4f} | "
            f"{s['KDE_NLL']:6.4f} {s['wKDE_NLL']:6.4f} | {fmt_opt(s['hard_wKDE_NLL'], '7.4f')} {fmt_opt(s['hard_ES_w'], '6.4f')}"
        )

    print(f"\n--- TABLE B ({fold_label}): Slot Collapse, Ranking & Kinematics vs B0 ---")
    hdr_b = f"{'Model':26s} | {'st_pNr':6s} {'st_#Nr':6s} {'st_H':5s} {'st_mxP':6s} | {'Top1':6s} {'cFDE3':6s} {'rFDE3':6s} | {'rFDE5':6s} | {'spd95%':7s} {'accM%':7s} {'jrk95%':7s} {'jAcc95%':7s}"
    print(hdr_b)
    print("-" * len(hdr_b))
    for s in results.values():
        mr = s["motion_vs_B0_pct"]
        print(
            f"{s['name']:26s} | {fmt_opt(s['st_p_near_02'], '6.3f')} {fmt_opt(s['st_n_near_02'], '6.2f')} "
            f"{fmt_opt(s['st_H_p'], '5.3f')} {fmt_opt(s['st_maxP'], '6.3f')} | "
            f"{s['Top1_FDE']:6.4f} {s['cFDE3']:6.4f} {s['rFDE3']:6.4f} | {s['rFDE5']:6.4f} | "
            f"{mr['speed_p95']:+6.2f}% {mr['accel_mean']:+6.2f}% {mr['jerk_p95']:+6.2f}% {mr['junction_accel_p95']:+6.2f}%"
        )

    # Table C: Mechanism diagnostics
    print(f"\n--- TABLE C ({fold_label}): MCUS Mechanism Interpretability ---")
    ob_w = np.asarray(mc_meta["overlap_bias_weight"], dtype=np.float64)
    print("Learned overlap_bias weights (4 heads x [close 0.35m, mid 1.00m]):")
    for h_i in range(ob_w.shape[0]):
        print(f"  Head {h_i}: w_close(0.35m)={ob_w[h_i, 0]:+.4f}, w_mid(1.00m)={ob_w[h_i, 1]:+.4f}")

    gate = mc_pred["gate"].astype(np.float64)
    ov = mc_pred["ov"].astype(np.float64)
    st_m = masks["stationary"].astype(bool)
    sd_m = masks["steady"].astype(bool)
    mv_m = ~st_m
    hd_m = masks["hard"].astype(bool)
    gate_diag = {}
    print(f"{'Subset / Overlap Slice':24s} | {'Count':8s} | {'gate_mean':9s} {'gate_p10':8s} {'gate_p90':8s} {'ov_mean':8s}")
    print("-" * 74)
    for label, g_vals, ov_vals in [
        ("overall", gate.reshape(-1), ov.reshape(-1)),
        ("stationary", gate[st_m].reshape(-1), ov[st_m].reshape(-1)),
        ("steady", gate[sd_m].reshape(-1), ov[sd_m].reshape(-1)),
        ("moving", gate[mv_m].reshape(-1), ov[mv_m].reshape(-1)),
        ("hard", gate[hd_m].reshape(-1), ov[hd_m].reshape(-1)),
        ("high_overlap (ov>0.5)", gate[ov > 0.5], ov[ov > 0.5]),
        ("low_overlap (ov<0.1)", gate[ov < 0.1], ov[ov < 0.1]),
    ]:
        if len(g_vals) == 0:
            gate_diag[label] = dict(count=0, mean=None, p10=None, p90=None, ov_mean=None)
            print(f"{label:24s} | {0:8d} | N/A")
        else:
            row = dict(
                count=int(len(g_vals)),
                mean=float(g_vals.mean()),
                p10=float(np.quantile(g_vals, 0.10)),
                p90=float(np.quantile(g_vals, 0.90)),
                ov_mean=float(ov_vals.mean()),
            )
            gate_diag[label] = row
            print(f"{label:24s} | {row['count']:8d} | {row['mean']:9.4f} {row['p10']:8.4f} {row['p90']:8.4f} {row['ov_mean']:8.4f}")
    return gate_diag


def evaluate_univ(test_data, masks: dict, device: str, seed: int = 42):
    results = {}
    # 1. B0
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

    # 2. O-conf+E0 & O-conf+QFL
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

    # 3. ES-Calib-1L & Decoupled-Calib (z_raw tau=1)
    es_pred = ES_PROBE_OUT / f"ES-Calib-1L_univ_s{seed}/dev_predictions.npz"
    es_meta = ES_PROBE_OUT / f"ES-Calib-1L_univ_s{seed}/TRAIN_META.json"
    if es_pred.exists():
        epz = np.load(es_pred, allow_pickle=True)
        em = json.loads(es_meta.read_text(encoding="utf-8"))
        results["ES-Calib-1L"] = evaluate_predictions(
            "ES-Calib-1L (Coupled)", epz["paths"], epz["logits"], test_data, masks,
            em["n_total"], em["n_train"], b0_motion=b0_mot,
        )

    dec_pred_p = DEC_PROBE_OUT / f"Decoupled-Calib-1L_univ_s{seed}/dev_predictions.npz"
    dec_meta_p = DEC_PROBE_OUT / f"Decoupled-Calib-1L_univ_s{seed}/TRAIN_META.json"
    if dec_pred_p.exists():
        dpz = np.load(dec_pred_p, allow_pickle=True)
        dm = json.loads(dec_meta_p.read_text(encoding="utf-8"))
        z_raw = dpz["logits"] * dpz["tau"][:, None]
        results["Decoupled-Calib-zraw"] = evaluate_predictions(
            "Decoupled-Calib (z_raw)", dpz["paths"], z_raw, test_data, masks,
            dm["n_total"], dm["n_train"], b0_motion=b0_mot,
        )

    # 4. MCUS-Decoupled-1L
    mc_dir = OUT_ROOT / f"MCUS-Decoupled-1L_univ_s{seed}"
    mc_pred = np.load(mc_dir / "dev_predictions.npz", allow_pickle=True)
    mc_meta = json.loads((mc_dir / "TRAIN_META.json").read_text(encoding="utf-8"))
    mc_s = evaluate_predictions(
        "MCUS-Decoupled (Ours L1)", mc_pred["paths"], mc_pred["logits"], test_data, masks,
        mc_meta["n_total"], mc_meta["n_train"], b0_motion=b0_mot,
    )
    mc_s["train_seconds"] = mc_meta["seconds"]
    results["MCUS-Decoupled-1L"] = mc_s
    results["MCUS-Decoupled-L0"] = evaluate_predictions(
        "MCUS-Decoupled (L0 aux)", mc_pred["aux_paths"], mc_pred["aux_logits"], test_data, masks,
        mc_meta["n_total"], mc_meta["n_train"], b0_motion=b0_mot,
    )

    gate_diag = print_tables(f"UNIV s{seed}, N={len(test_data['ids'])}", results, mc_pred, mc_meta, masks)
    return dict(results=results, gate_diagnostics=gate_diag, overlap_bias_weight=mc_meta["overlap_bias_weight"])


def evaluate_eth(test_data, masks: dict, device: str, seed: int = 42):
    results = {}
    preds_map = {}

    # 1. B0_eth_s42
    b0_ck = QROOT / "e_a_diag/results/weights/B0_eth_s42.pt"
    m_b0, _ = core.make_model(device, seed)
    m_b0.load_state_dict(torch.load(b0_ck, map_location="cpu", weights_only=True), strict=True)
    m_b0.requires_grad_(False)
    b0_n = sum(p.numel() for p in m_b0.parameters())
    b0_p, b0_l = infer_b0(m_b0, test_data)
    b0_sum = evaluate_predictions("B0 (Baseline)", b0_p, b0_l, test_data, masks, b0_n, b0_n, b0_motion=None)
    b0_sum["motion_vs_B0_pct"] = {k: 0.0 for k in b0_sum["motion"]}
    verify_against_saved(b0_sum, REV_RUNS / "B0_eth_s42/dev_metrics.npz", "B0_eth_s42")
    results["B0"] = b0_sum
    preds_map["B0"] = (b0_p, b0_l)
    b0_mot = b0_sum["motion"]
    del m_b0

    # 2. O-conf+E0 & O-conf+QFL
    for arm in ("O-conf+E0", "O-conf+QFL"):
        ck = QROOT / "frozen_replay/results/weights" / f"{arm}_eth_s{seed}.pt"
        m_q, _ = core.make_model(device, seed)
        attach_qrank(m_q, arm)
        m_q.decoder.use_kin = True
        m_q.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
        m_q.requires_grad_(False)
        q_n = sum(p.numel() for p in m_q.parameters())
        qp, ql = infer_qrank(m_q, test_data)
        qs = evaluate_predictions(f"{arm} (3L)", qp, ql, test_data, masks, q_n, q_n, b0_motion=b0_mot)
        verify_against_saved(qs, REV_RUNS / f"{arm}_eth_s{seed}/dev_metrics.npz", f"{arm}_eth_s{seed}")
        results[arm] = qs
        preds_map[arm] = (qp, ql)
        del m_q

    # 3. MCUS-Decoupled-1L
    mc_dir = OUT_ROOT / f"MCUS-Decoupled-1L_eth_s{seed}"
    mc_pred = np.load(mc_dir / "dev_predictions.npz", allow_pickle=True)
    mc_meta = json.loads((mc_dir / "TRAIN_META.json").read_text(encoding="utf-8"))
    mc_s = evaluate_predictions(
        "MCUS-Decoupled (Ours L1)", mc_pred["paths"], mc_pred["logits"], test_data, masks,
        mc_meta["n_total"], mc_meta["n_train"], b0_motion=b0_mot,
    )
    mc_s["train_seconds"] = mc_meta["seconds"]
    results["MCUS-Decoupled-1L"] = mc_s
    results["MCUS-Decoupled-L0"] = evaluate_predictions(
        "MCUS-Decoupled (L0 aux)", mc_pred["aux_paths"], mc_pred["aux_logits"], test_data, masks,
        mc_meta["n_total"], mc_meta["n_train"], b0_motion=b0_mot,
    )
    preds_map["MCUS-Decoupled-1L"] = (mc_pred["paths"], mc_pred["logits"])
    preds_map["MCUS-Decoupled-L0"] = (mc_pred["aux_paths"], mc_pred["aux_logits"])

    gate_diag = print_tables(f"ETH s{seed}, N={len(test_data['ids'])}", results, mc_pred, mc_meta, masks)

    # ETH High-Speed Slice (> 1.3 m/s) vs Rest
    obs_speed = np.linalg.norm(np.diff(test_data["past"][:, 0].astype(np.float64), axis=1), axis=-1).mean(axis=1) / DT
    fast_m = obs_speed > 1.3
    rest_m = ~fast_m
    gt = test_data["future"].astype(np.float64)
    mc_gate = mc_pred["gate"].astype(np.float64)
    slice_diag = {}
    print(f"\n--- TABLE C2 (ETH Speed Slices): Fast (>1.3 m/s, N={fast_m.sum()}) vs Rest (N={rest_m.sum()}) ---")
    print(f"{'Model':22s} | {'Slice':10s} | {'minFDE':6s} {'Top1':6s} {'ES_w':6s} {'wKDE':6s} {'rFDE3':6s} {'gate':6s}")
    print("-" * 76)
    for mname, (P, Z) in preds_map.items():
        slice_diag[mname] = {}
        for sname, msk in (("fast>1.3", fast_m), ("rest<=1.3", rest_m)):
            Pm, Zm, gtm = P[msk].astype(np.float64), Z[msk].astype(np.float64), gt[msk]
            fm = forecast_metrics(Pm, Zm, gtm)
            wk = kde_nll_vec(Pm, gtm, weights=softmax(Zm))
            fde = np.linalg.norm(Pm[:, :, -1] - gtm[:, None, -1], axis=-1)
            r_ord = np_risk_order(Pm[:, :, -1], softmax(Zm))
            rfde3 = float(np.take_along_axis(fde, r_ord, axis=1)[:, :3].min(1).mean())
            gm = float(mc_gate[msk].mean()) if "MCUS" in mname else None
            row = dict(
                N=int(msk.sum()),
                minFDE=float(fm["minFDE"].mean()),
                Top1_FDE=float(fm["Top1_FDE"].mean()),
                ES_w=float(fm["ES"].mean()),
                wKDE_NLL=float(wk.mean()),
                rFDE3=rfde3,
                gate_mean=gm,
            )
            slice_diag[mname][sname] = row
            print(
                f"{mname:22s} | {sname:10s} | {row['minFDE']:6.4f} {row['Top1_FDE']:6.4f} "
                f"{row['ES_w']:6.4f} {row['wKDE_NLL']:6.4f} {row['rFDE3']:6.4f} {fmt_opt(gm, '6.4f')}"
            )

    return dict(results=results, gate_diagnostics=gate_diag, speed_slices=slice_diag,
                overlap_bias_weight=mc_meta["overlap_bias_weight"])


def evaluate_c0(test_data, masks: dict, device: str, seed: int = 42):
    results = {}
    # Load sealed C0 B0_s42, S2_s42, S4_s42
    b0_z = np.load(C0_RUNS / f"B0_s{seed}/dev_predictions.npz", allow_pickle=True)
    b0_sum = evaluate_predictions("B0 (Baseline)", b0_z["paths"], b0_z["logits"], test_data, masks, 295474, 295474, b0_motion=None)
    b0_sum["motion_vs_B0_pct"] = {k: 0.0 for k in b0_sum["motion"]}
    verify_against_saved(b0_sum, C0_RUNS / f"B0_s{seed}/dev_metrics.npz", f"C0 B0_s{seed}")
    results["B0"] = b0_sum
    b0_mot = b0_sum["motion"]
    for arm, params in (("S2", 358789), ("S4", 358789)):
        az = np.load(C0_RUNS / f"{arm}_s{seed}/dev_predictions.npz", allow_pickle=True)
        asum = evaluate_predictions(f"{arm} (C0 1L)", az["paths"], az["logits"], test_data, masks, params, params, b0_motion=b0_mot)
        verify_against_saved(asum, C0_RUNS / f"{arm}_s{seed}/dev_metrics.npz", f"C0 {arm}_s{seed}")
        results[arm] = asum

    mc_dir = OUT_ROOT / f"MCUS-Decoupled-1L_C0_s{seed}"
    mc_pred = np.load(mc_dir / "dev_predictions.npz", allow_pickle=True)
    mc_meta = json.loads((mc_dir / "TRAIN_META.json").read_text(encoding="utf-8"))
    mc_s = evaluate_predictions(
        "MCUS-Decoupled (Ours L1)", mc_pred["paths"], mc_pred["logits"], test_data, masks,
        mc_meta["n_total"], mc_meta["n_train"], b0_motion=b0_mot,
    )
    mc_s["train_seconds"] = mc_meta["seconds"]
    results["MCUS-Decoupled-1L"] = mc_s
    results["MCUS-Decoupled-L0"] = evaluate_predictions(
        "MCUS-Decoupled (L0 aux)", mc_pred["aux_paths"], mc_pred["aux_logits"], test_data, masks,
        mc_meta["n_total"], mc_meta["n_train"], b0_motion=b0_mot,
    )
    gate_diag = print_tables(f"C0 s{seed}, N={len(test_data['ids'])}", results, mc_pred, mc_meta, masks)
    return dict(results=results, gate_diagnostics=gate_diag, overlap_bias_weight=mc_meta["overlap_bias_weight"])


def bench_latency(test_data, seed: int = 42):
    print("\n--- TABLE D: Real Inference Latency (B0 vs MCUS-Decoupled-1L) ---")
    lat = {}
    ckpt = OUT_ROOT / f"MCUS-Decoupled-1L_univ_s{seed}/final_ema.pt"
    for dev in ("cuda:0", "cpu"):
        for bs in (1, 64):
            iters = 100 if dev == "cuda:0" else (50 if bs == 1 else 20)
            warm = 20 if bs == 1 else 10
            p = torch.as_tensor(test_data["past"][:bs], device=dev)
            n = torch.as_tensor(test_data["num_valid"][:bs], device=dev)
            for label, builder in (("B0", lambda d: core.make_model(d, seed)[0]),
                                   ("MCUS-Decoupled-1L", None)):
                if label == "B0":
                    m = builder(dev)
                else:
                    m, _ = core.make_model(dev, seed)
                    attach_mcus(m)
                    if ckpt.exists():
                        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
                m.eval()
                with torch.no_grad():
                    for _ in range(warm):
                        core.prediction(m, p, n)
                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        core.prediction(m, p, n)
                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) / iters * 1000.0
                params = sum(x.numel() for x in m.parameters())
                key = f"{label}_{dev}_bs{bs}"
                lat[key] = dict(params=params, ms_per_batch=ms, ms_per_agent=ms / bs)
                print(f"  {label:20s} | {dev:6s} | bs={bs:2d} | params={params:7d} | {ms:7.3f} ms/batch ({ms/bs:.4f} ms/agent)")
                del m
    return lat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train_eval_univ", "run_two_scenes"), required=True)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-min", type=float, default=30.0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda:0"
    print(f"Device: {torch.cuda.get_device_name(0)}, torch={torch.__version__}", flush=True)

    (u_tr_p, u_tr_n), (u_te_p, u_te_n) = inputs("univ")
    u_fit = load_data(u_tr_p, u_tr_n)
    u_test = load_data(u_te_p, u_te_n)
    u_masks = groups(u_test)

    if args.mode == "smoke":
        run_smoke(u_fit, device)
        return

    t_wall0 = time.monotonic()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}

    # Scene 1: univ
    u_dir = train_fold("univ", u_fit, u_test, args.epochs, args.seed, device)
    u_meta = json.loads((u_dir / "TRAIN_META.json").read_text(encoding="utf-8"))
    summary["univ"] = evaluate_univ(u_test, u_masks, device, args.seed)
    summary["latency"] = bench_latency(u_test, args.seed)

    if args.mode == "run_two_scenes":
        elapsed = time.monotonic() - t_wall0
        sec_per_step = u_meta["seconds"] / max(args.epochs * u_meta["steps_per_epoch"], 1)
        est_eth_sec = sec_per_step * 474 * args.epochs
        est_c0_sec = sec_per_step * 340 * args.epochs
        cap_sec = (args.budget_min + 4.0) * 60.0
        print(
            f"\n[BUDGET] elapsed={elapsed/60:.2f}m, sec/step={sec_per_step*1000:.2f}ms, "
            f"est_eth={est_eth_sec/60:.2f}m, est_c0={est_c0_sec/60:.2f}m, cap={cap_sec/60:.1f}m",
            flush=True,
        )
        if elapsed + est_eth_sec <= cap_sec:
            (e_tr_p, e_tr_n), (e_te_p, e_te_n) = inputs("eth")
            e_fit = load_data(e_tr_p, e_tr_n)
            e_test = load_data(e_te_p, e_te_n)
            e_masks = groups(e_test)
            train_fold("eth", e_fit, e_test, args.epochs, args.seed, device)
            summary["eth"] = evaluate_eth(e_test, e_masks, device, args.seed)
            summary["scene2"] = "eth"
        elif elapsed + est_c0_sec <= cap_sec:
            (c_tr_p, c_tr_n), (c_te_p, c_te_n) = inputs("C0")
            c_fit = load_data(c_tr_p, c_tr_n)
            c_test = load_data(c_te_p, c_te_n)
            c_masks = groups(c_test)
            train_fold("C0", c_fit, c_test, args.epochs, args.seed, device)
            summary["C0"] = evaluate_c0(c_test, c_masks, device, args.seed)
            summary["scene2"] = "C0"
        else:
            summary["scene2"] = "SKIPPED_OVER_BUDGET"
            print("[BUDGET] Neither eth nor C0 fits within budget; keeping univ summary.", flush=True)

    summary["total_wall_seconds"] = time.monotonic() - t_wall0
    out_json = OUT_ROOT / "RESULT_SUMMARY.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary -> {out_json} (total wall={summary['total_wall_seconds']/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
