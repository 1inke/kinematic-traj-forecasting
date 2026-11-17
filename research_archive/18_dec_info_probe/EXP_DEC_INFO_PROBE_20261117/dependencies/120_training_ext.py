"""E4 training-method factors (torch; NOT executed in authoring env).

Each factor replaces exactly one piece of baseline_factory.loss_value / augmentation;
everything else (ModeQueryLoss terms, 0.2 soft-coverage, optimizer, EMA, schedule) is kept.

T1  annealed soft-WTA regression   (only the trajectory-regression term changes)
T2  change-weighted loss           (training-only GT weight; never a model input)
T3  future-only turn-onset bend    (augmentation of the focal FUTURE only)
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

# ------------------------------------------------------------------ T1
TAU0, TAU1, HARD_FROM = 0.30, 0.02, 0.50   # metres; switch to exact hard WTA after 50% of updates


def tau_at(progress: float) -> float | None:
    """progress = update / total_updates in [0,1]; None means hard WTA (exact B0 term)."""
    if progress >= HARD_FROM:
        return None
    u = progress / HARD_FROM
    return TAU1 + (TAU0 - TAU1) * 0.5 * (1 + math.cos(math.pi * u))


def loss_value_T1(pred, target, logits, criterion, progress):
    """Same as baseline_factory.loss_value but the winner-only regression term is replaced
    by a responsibility-weighted one while tau is not None."""
    base, detail = criterion(pred, target, logits)
    distances = torch.linalg.vector_norm(pred - target.unsqueeze(-3), dim=-1)          # [B,1,K,T]
    quality = distances.mean(-1) + criterion.winner_fde_weight * distances[..., -1]
    tau = tau_at(progress)
    if tau is not None:
        per_k = criterion.regression.criterion(pred, target.unsqueeze(-3).expand_as(pred)).sum(-1)
        per_k = (per_k * criterion.regression.weights).mean(-1)                         # [B,1,K]
        r = torch.softmax(-quality.detach() / tau, -1)
        soft_reg = (r * per_k).sum(-1).mean()
        # remove the hard winner term (recomputed with graph, numerically identical) and add the soft one
        base = base - _regression_live(pred, target, criterion, quality) + soft_reg
        detail = dict(detail, regression_soft=soft_reg.detach(), tau=tau)
    responsibility = torch.softmax(-(distances.mean(-1) + .25 * distances[..., -1]).detach() / .10, -1)
    coverage = (responsibility * (distances.mean(-1) + .25 * distances[..., -1])).sum(-1).mean()
    return base + .20 * coverage, detail


def _regression_live(pred, target, criterion, quality):
    """Recompute ModeQueryLoss's winner regression WITH graph so it can be removed exactly."""
    winner = quality.detach().argmin(-1)
    idx = winner[..., None, None, None].expand(*winner.shape, 1, pred.shape[-2], pred.shape[-1])
    return criterion.regression(pred.gather(-3, idx).squeeze(-3), target)


# ------------------------------------------------------------------ T2
LAMBDA_CHANGE = 1.0


def change_score(past_focal, future_focal):
    """Same formula as the C0 grouping score (REGIME_CHANGE_DETECTOR future_change_score);
    rotation/translation invariant, so it can be computed on raw world coordinates."""
    observed = past_focal[:, -1] - past_focal[:, -2]
    future_mean = (future_focal[:, -1] - past_focal[:, -1]) / float(future_focal.shape[-2])
    os_, fs = observed.norm(dim=-1), future_mean.norm(dim=-1)
    prod = os_ * fs
    cos = torch.where(prod > 1e-6, ((observed * future_mean).sum(-1) / prod.clamp_min(1e-6)).clamp(-1, 1),
                      torch.ones_like(prod))
    turn = ((1 - cos) / 2).clamp(0, 1)
    speed = (torch.abs(torch.log((fs + .05) / (os_ + .05))) / math.log(2.0)).clamp(0, 1)
    return torch.maximum(turn, speed)


def change_weights(past_world, future_world):
    w = 1.0 + LAMBDA_CHANGE * change_score(past_world[:, 0], future_world)
    return w / w.mean()


def loss_value_T2(pred, target, logits, criterion, weights):
    base, detail = criterion(pred, target, logits, sample_weight=weights)
    distances = torch.linalg.vector_norm(pred - target.unsqueeze(-3), dim=-1)
    quality = distances.mean(-1) + .25 * distances[..., -1]
    responsibility = torch.softmax(-quality.detach() / .10, -1)
    coverage = ((responsibility * quality).sum(-1).reshape(len(weights), -1).mean(-1) * weights).mean()
    return base + .20 * coverage, detail


# ------------------------------------------------------------------ T3
KAPPA_MAX, P_BEND = 0.03, 0.30   # rad per step (max ~0.36 rad = 21 deg at step 12); probability


def bend_future_only(past, future, generator=None):
    """Rotate the focal future about the last observed point by kappa*t (t=1..12).
    past is untouched -> creates an unobservable turn onset at t=0.  Applied BEFORE
    baseline_factory.prepare(); future here is [B,12,2] world."""
    b = future.shape[0]
    active = torch.rand(b, generator=generator, device=future.device) < P_BEND
    kappa = (torch.rand(b, generator=generator, device=future.device) * 2 - 1) * KAPPA_MAX * active
    t = torch.arange(1, future.shape[1] + 1, device=future.device, dtype=future.dtype)
    ang = kappa[:, None] * t[None]
    c, s = ang.cos(), ang.sin()
    p0 = past[:, 0, -1:, :]
    rel = future - p0
    x = c * rel[..., 0] - s * rel[..., 1]
    y = s * rel[..., 0] + c * rel[..., 1]
    return torch.stack((x, y), -1) + p0, kappa
