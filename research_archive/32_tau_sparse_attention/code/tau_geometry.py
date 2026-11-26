"""Disc collision time, degree-constrained selection, and F16 energy bias."""

import torch
from torch import nn


def collision_time(position, velocity, radius):
    a = velocity.square().sum(-1)
    b = -(position * velocity).sum(-1)
    c = position.square().sum(-1) - radius**2
    discriminant = b.square() - a * c
    positive_root = (a > 1e-12) & (b > 0) & (c > 0) & (discriminant >= 0)
    root = c / (b + discriminant.clamp_min(0).sqrt()).clamp_min(1e-12)
    return torch.where(c <= 0, 0.0, torch.where(positive_root, root, torch.inf))


def history(past, frequency, radius):
    velocity = torch.zeros_like(past)
    velocity[:, 1:] = (past[:, 1:] - past[:, :-1]) * frequency
    if past.shape[1] > 1:
        velocity[:, 0] = velocity[:, 1]
    return collision_time(
        past[:, :, None] - past[:, :, :, None],
        velocity[:, :, None] - velocity[:, :, :, None],
        radius,
    )


def select(tau, distance, candidates, cutoff, k_min, k_max):
    """Rank collision neighbors first, then fill the minimum by distance."""
    close = (tau <= cutoff) & candidates
    # Stable lexicographic order: collision status/time, then distance, then slot.
    distance_order = distance.masked_fill(~candidates, torch.inf).argsort(dim=-1, stable=True)
    priority = tau.masked_fill(~close, torch.inf).gather(-1, distance_order)
    order = distance_order.gather(-1, priority.argsort(dim=-1, stable=True))
    count = torch.minimum(close.sum(-1).clamp(min=k_min, max=k_max), candidates.sum(-1))
    selected = torch.arange(tau.shape[-1], device=tau.device) < count.unsqueeze(-1)
    return torch.zeros_like(candidates).scatter(-1, order, selected)


def energy(tau, power):
    # Units chosen so E(0.4 s)=1. The floor bounds overlap and near-zero times.
    time = tau.clamp_min(0.2)
    return (0.4 / time).pow(power) * torch.exp((0.4 - time) / 3.0)


def attach(model, arm, power):
    encoder = model.agent_aware_trans_encoder
    if arm in ("T1", "T2"):
        model.graph_builder.tau_radius = 0.6
    if arm in ("T2", "T3"):
        assert encoder.heads == 4 and encoder.use_temporal_social_cross
        encoder.tau_beta = nn.Parameter(next(model.parameters()).new_zeros(encoder.heads))
        encoder.tau_power = float(power)
