"""BASE 直接位置；KIN 积分 + 软限幅。"""
from __future__ import annotations
import torch
from torch import nn
from .integrate import H, initial_velocity_from_past, integrate_acceleration

K = 20
HIDDEN = 128
QUERY_DIM = 16


class SharedTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.past_mlp = nn.Sequential(
            nn.Linear(8 * 2 + 2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU()
        )
        self.queries = nn.Parameter(torch.randn(K, QUERY_DIM) * 0.02)
        self.fuse = nn.Sequential(
            nn.Linear(HIDDEN + QUERY_DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU()
        )

    def encode(self, past: torch.Tensor) -> torch.Tensor:
        v0 = initial_velocity_from_past(past)
        x = torch.cat([past.reshape(past.shape[0], -1), v0], dim=-1)
        h = self.past_mlp(x)
        q = self.queries[None].expand(past.shape[0], -1, -1)
        h = h[:, None, :].expand(-1, K, -1)
        return self.fuse(torch.cat([h, q], dim=-1))


class DirectPositionDecoder(nn.Module):
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


class KinematicClampDecoder(nn.Module):
    def __init__(self, v_max: float, a_max: float):
        super().__init__()
        self.trunk = SharedTrunk()
        self.head = nn.Linear(HIDDEN, H * 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.v_max = float(v_max)
        self.a_max = float(a_max)

    def forward(self, past: torch.Tensor):
        z = self.trunk.encode(past)
        raw = self.head(z).view(-1, K, H, 6)
        v_hat, a_hat, dp_hat = raw[..., 0:2], raw[..., 2:4], raw[..., 4:6]
        v0 = initial_velocity_from_past(past)
        paths, v_int, dp_int, a_used = integrate_acceleration(
            a_hat, v0, v_max=self.v_max, a_max=self.a_max
        )
        return dict(
            paths=paths,
            arm="kinematic_clamp",
            v_hat=v_hat,
            a_hat=a_hat,
            dp_hat=dp_hat,
            v_int=v_int,
            dp_int=dp_int,
            a_used=a_used,
            v_max=self.v_max,
            a_max=self.a_max,
        )


def build(arm: str, v_max: float = 1.27, a_max: float = 0.57) -> nn.Module:
    if arm in ("BASE", "direct_position"):
        return DirectPositionDecoder()
    if arm in ("KIN", "kinematic_clamp"):
        return KinematicClampDecoder(v_max=v_max, a_max=a_max)
    raise ValueError(arm)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
