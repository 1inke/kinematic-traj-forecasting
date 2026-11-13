"""离散运动学积分（KIN 臂唯一轨迹定义）。"""
from __future__ import annotations
import torch

DT = 0.4
H = 12


def initial_velocity_from_past(past: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """past: (B, 8, 2), 末点应为原点。v0 = (p0 - p_{-1}) / dt。"""
    return (past[:, -1] - past[:, -2]) / dt


def integrate_acceleration(
    a: torch.Tensor,
    v0: torch.Tensor,
    p0: torch.Tensor | None = None,
    dt: float = DT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """a: (B,K,H,2) 或 (B,H,2)；v0/p0: (B,2)。

    递推：
      v_t = v_{t-1} + dt * a_t
      dp_t = dt * v_t
      p_t = p_{t-1} + dp_t

    返回 paths(B,K,H,2), v(B,K,H,2), dp(B,K,H,2)
    """
    if a.dim() == 3:
        a = a.unsqueeze(1)
    B, K, T, D = a.shape
    assert T == H and D == 2
    if p0 is None:
        p0 = torch.zeros(B, 2, device=a.device, dtype=a.dtype)
    v = v0[:, None, :].expand(B, K, 2).to(dtype=a.dtype)
    p = p0[:, None, :].expand(B, K, 2).to(dtype=a.dtype)
    ps, vs, dps = [], [], []
    for t in range(T):
        v = v + dt * a[:, :, t]
        dp = dt * v
        p = p + dp
        ps.append(p)
        vs.append(v)
        dps.append(dp)
    return torch.stack(ps, 2), torch.stack(vs, 2), torch.stack(dps, 2)


def constant_accel_closed_form(a: torch.Tensor, v0: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """恒加速度解析位置，用于单测。a:(B,2) 常量重复 H 步。

    与 integrate 递推一致：p_t = t*dt*v0 + dt^2 * a * t*(t+1)/2
    """
    t = torch.arange(1, H + 1, device=a.device, dtype=a.dtype)
    return (t[None, :, None] * dt) * v0[:, None, :] + (dt ** 2) * a[:, None, :] * (
        t[None, :, None] * (t[None, :, None] + 1) / 2.0
    )
