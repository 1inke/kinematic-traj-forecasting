"""多模式潜变量 → 运动学积分；多样性用 endpoint hinge，不用全员同 GT。"""
from __future__ import annotations
import torch
from torch import nn
from .integrate import H, initial_velocity_from_past, integrate_acceleration

K = 20
N_MODES = 4
HIDDEN = 128


class ModeDiverseKinDecoder(nn.Module):
    def __init__(self, v_max: float, a_max: float, n_modes: int = N_MODES, k: int = K):
        super().__init__()
        self.k = k
        self.n_modes = n_modes
        self.per_mode = k // n_modes
        assert k % n_modes == 0
        self.past_mlp = nn.Sequential(
            nn.Linear(8 * 2 + 2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU()
        )
        self.mode_embed = nn.Parameter(torch.randn(n_modes, HIDDEN) * 0.02)
        self.slot_embed = nn.Parameter(torch.randn(self.per_mode, 32) * 0.02)
        self.fuse = nn.Sequential(
            nn.Linear(HIDDEN + HIDDEN + 32, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU()
        )
        self.head = nn.Linear(HIDDEN, H * 2)  # acceleration only
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.v_max = float(v_max)
        self.a_max = float(a_max)

    def forward(self, past: torch.Tensor):
        B = past.shape[0]
        v0 = initial_velocity_from_past(past)
        h = self.past_mlp(torch.cat([past.reshape(B, -1), v0], dim=-1))  # B,H
        zs = []
        for m in range(self.n_modes):
            me = self.mode_embed[m][None].expand(B, -1)
            for s in range(self.per_mode):
                se = self.slot_embed[s][None].expand(B, -1)
                zs.append(self.fuse(torch.cat([h, me, se], dim=-1)))
        z = torch.stack(zs, dim=1)  # B,K,H
        a = self.head(z).view(B, self.k, H, 2)
        paths, v_int, dp_int, a_used = integrate_acceleration(
            a, v0, v_max=self.v_max, a_max=self.a_max
        )
        return dict(paths=paths, arm="mode_diverse_kin", a_hat=a, v_int=v_int, dp_int=dp_int, a_used=a_used)


def build(arm: str, **kw):
    if arm in ("MDK", "mode_diverse_kin"):
        return ModeDiverseKinDecoder(v_max=kw["v_max"], a_max=kw["a_max"])
    raise ValueError(arm)


def count_params(net):
    return sum(p.numel() for p in net.parameters())
