"""两臂共享 trunk；BASE 直接位置，KIN 输出 (v,a,dp) 再积分。"""
from __future__ import annotations
import torch
from torch import nn
from .integrate import DT, H, initial_velocity_from_past, integrate_acceleration

K = 20
HIDDEN = 128
QUERY_DIM = 16


class SharedTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.past_mlp = nn.Sequential(
            nn.Linear(8 * 2 + 2, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.randn(K, QUERY_DIM) * 0.02)
        self.fuse = nn.Sequential(
            nn.Linear(HIDDEN + QUERY_DIM, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
        )

    def encode(self, past: torch.Tensor) -> torch.Tensor:
        # past (B,8,2); append v0
        v0 = initial_velocity_from_past(past)
        x = torch.cat([past.reshape(past.shape[0], -1), v0], dim=-1)
        h = self.past_mlp(x)
        q = self.queries[None].expand(past.shape[0], -1, -1)
        h = h[:, None, :].expand(-1, K, -1)
        return self.fuse(torch.cat([h, q], dim=-1))  # (B,K,HIDDEN)


class DirectPositionDecoder(nn.Module):
    """BASE：直接输出相对末观测（原点）的位置。"""

    def __init__(self):
        super().__init__()
        self.trunk = SharedTrunk()
        self.head = nn.Linear(HIDDEN, H * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, past: torch.Tensor):
        z = self.trunk.encode(past)
        p = self.head(z).view(-1, K, H, 2)
        return dict(paths=p, arm="direct_position")


class KinematicDecoder(nn.Module):
    """KIN：每步预测 (v̂, â, Δp̂)，轨迹仅由 â 积分得到。"""

    def __init__(self):
        super().__init__()
        self.trunk = SharedTrunk()
        self.head = nn.Linear(HIDDEN, H * 6)  # v,a,dp per step
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, past: torch.Tensor):
        z = self.trunk.encode(past)
        raw = self.head(z).view(-1, K, H, 6)
        v_hat = raw[..., 0:2]
        a_hat = raw[..., 2:4]
        dp_hat = raw[..., 4:6]
        v0 = initial_velocity_from_past(past)
        paths, v_int, dp_int = integrate_acceleration(a_hat, v0)
        return dict(
            paths=paths,
            arm="kinematic_integration",
            v_hat=v_hat,
            a_hat=a_hat,
            dp_hat=dp_hat,
            v_int=v_int,
            dp_int=dp_int,
        )


def build(arm: str) -> nn.Module:
    if arm in ("BASE", "direct_position"):
        return DirectPositionDecoder()
    if arm in ("KIN", "kinematic_integration"):
        return KinematicDecoder()
    raise ValueError(arm)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
