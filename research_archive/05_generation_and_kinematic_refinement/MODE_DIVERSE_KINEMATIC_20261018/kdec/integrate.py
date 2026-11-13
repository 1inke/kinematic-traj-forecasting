"""离散运动学积分 + 可微软限幅。"""
from __future__ import annotations
import torch

DT = 0.4
H = 12


def initial_velocity_from_past(past: torch.Tensor, dt: float = DT) -> torch.Tensor:
    return (past[:, -1] - past[:, -2]) / dt


def soft_clamp_norm(x: torch.Tensor, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """可微：若 ||x|| > max_norm 则缩放到 max_norm。max_norm 可为标量或 broadcastable。"""
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    if not torch.is_tensor(max_norm):
        max_norm = torch.tensor(max_norm, device=x.device, dtype=x.dtype)
    scale = torch.clamp(max_norm / n, max=1.0)
    return x * scale


def integrate_acceleration(
    a: torch.Tensor,
    v0: torch.Tensor,
    p0: torch.Tensor | None = None,
    dt: float = DT,
    v_max: float | None = None,
    a_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 paths, v, dp, a_used。若给定 v_max/a_max 则软限幅。"""
    if a.dim() == 3:
        a = a.unsqueeze(1)
    B, K, T, D = a.shape
    assert T == H and D == 2
    if p0 is None:
        p0 = torch.zeros(B, 2, device=a.device, dtype=a.dtype)
    v = v0[:, None, :].expand(B, K, 2).to(dtype=a.dtype).clone()
    p = p0[:, None, :].expand(B, K, 2).to(dtype=a.dtype).clone()
    ps, vs, dps, aused = [], [], [], []
    for t in range(T):
        at = a[:, :, t]
        if a_max is not None:
            at = soft_clamp_norm(at, a_max)
        v = v + dt * at
        if v_max is not None:
            v = soft_clamp_norm(v, v_max)
        dp = dt * v
        p = p + dp
        ps.append(p)
        vs.append(v)
        dps.append(dp)
        aused.append(at)
    return torch.stack(ps, 2), torch.stack(vs, 2), torch.stack(dps, 2), torch.stack(aused, 2)
