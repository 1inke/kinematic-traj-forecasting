"""RES: 在原生 K20 上加有界运动学残差；IDENTITY: 原样；KIN_REF: 纯再生参考。"""
from __future__ import annotations
import torch
from torch import nn
from .integrate import H, initial_velocity_from_past, integrate_acceleration, soft_clamp_norm

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


class NativeAnchoredResidualDecoder(nn.Module):
    """paths = native + integrate(tanh(a)*a_res_max) with v0=0 (纯位移残差)。"""

    def __init__(self, a_res_max: float = 0.25, v_max: float = 1.27, a_max: float = 0.57):
        super().__init__()
        self.trunk = SharedTrunk()
        self.native_mlp = nn.Sequential(
            nn.Linear(H * 2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN)
        )
        self.fuse2 = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU()
        )
        self.head = nn.Linear(HIDDEN, H * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.a_res_max = float(a_res_max)
        self.v_max = float(v_max)
        self.a_max = float(a_max)

    def forward(self, past: torch.Tensor, native: torch.Tensor):
        # native: B,K,H,2
        B, Kk, Hh, _ = native.shape
        assert Kk == K and Hh == H
        z = self.trunk.encode(past)
        nfeat = self.native_mlp(native.reshape(B, K, -1))
        z = self.fuse2(torch.cat([z, nfeat], dim=-1))
        a_raw = self.head(z).view(B, K, H, 2)
        a_res = torch.tanh(a_raw) * self.a_res_max
        # 残差从 0 速度积分 → 位移残差
        v0 = torch.zeros(B, 2, device=past.device, dtype=past.dtype)
        dpaths, v_int, dp_int, a_used = integrate_acceleration(
            a_res, v0, v_max=None, a_max=self.a_res_max
        )
        paths = native + dpaths
        # 可选：对逐步速度做软提示（不改 path，仅审计）
        return dict(
            paths=paths,
            arm="native_residual",
            a_res=a_res,
            dpaths=dpaths,
            v_int=v_int,
            dp_int=dp_int,
            a_used=a_used,
            a_res_max=self.a_res_max,
            native=native,
        )


class IdentityNative(nn.Module):
    def forward(self, past: torch.Tensor, native: torch.Tensor):
        return dict(paths=native, arm="identity")


class KinematicClampDecoder(nn.Module):
    """纯再生参考（不吃 native）。"""

    def __init__(self, v_max: float, a_max: float):
        super().__init__()
        self.trunk = SharedTrunk()
        self.head = nn.Linear(HIDDEN, H * 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.v_max = float(v_max)
        self.a_max = float(a_max)

    def forward(self, past: torch.Tensor, native: torch.Tensor | None = None):
        z = self.trunk.encode(past)
        raw = self.head(z).view(-1, K, H, 6)
        v_hat, a_hat, dp_hat = raw[..., 0:2], raw[..., 2:4], raw[..., 4:6]
        v0 = initial_velocity_from_past(past)
        paths, v_int, dp_int, a_used = integrate_acceleration(
            a_hat, v0, v_max=self.v_max, a_max=self.a_max
        )
        return dict(
            paths=paths, arm="kinematic_clamp",
            v_hat=v_hat, a_hat=a_hat, dp_hat=dp_hat,
            v_int=v_int, dp_int=dp_int, a_used=a_used,
            v_max=self.v_max, a_max=self.a_max,
        )


def build(arm: str, **kw) -> nn.Module:
    if arm in ("RES", "native_residual"):
        return NativeAnchoredResidualDecoder(
            a_res_max=kw.get("a_res_max", 0.25),
            v_max=kw.get("v_max", 1.27),
            a_max=kw.get("a_max", 0.57),
        )
    if arm in ("ID", "identity"):
        return IdentityNative()
    if arm in ("KIN", "kinematic_clamp"):
        return KinematicClampDecoder(v_max=kw["v_max"], a_max=kw["a_max"])
    raise ValueError(arm)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
