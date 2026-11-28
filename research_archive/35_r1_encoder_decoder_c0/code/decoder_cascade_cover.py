"""Layer-differentiated cascade assignment and cover-order discriminative decoder (M0..M6, JM4)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from advisor_model import CrossLayer, set_valid, zero
from encoder_c1c2 import configure_encoder

DECODER_ARMS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "JM4")

# (l1_kind, theta0, m0, theta1, m1, disc_order, zero_redundancy, use_c2)
ARM_SPECS = {
    "M0":  ("none",    None, None, None, None, None,    False, False),
    "M1":  ("narrow",  0.5,  1,    0.5,  1,    None,    False, False),
    "M2":  ("narrow",  1.0,  3,    0.7,  2,    None,    False, False),
    "M3":  ("narrow",  1.0,  3,    0.7,  2,    "conf",  False, False),
    "M4":  ("narrow",  1.0,  3,    0.7,  2,    "cover", False, False),
    "M5":  ("narrow",  1.0,  3,    0.7,  2,    "cover", True,  False),
    "M6":  ("widened", 0.5,  1,    0.5,  1,    None,    False, False),
    "JM4": ("narrow",  1.0,  3,    0.7,  2,    "cover", False, True),
}

SHIFT = 0.30
THETA_D = 0.50
TAU_U = 1.0
SIGMA_O = 0.35
FIRST_D_PREV = 2.0
NUM_SINKS = 4


@torch.no_grad()
def cover_order_torch(
    end: Tensor,
    w: Tensor,
    first_argmax: bool = True,
    logits: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Greedy weighted k-median cover ordering in float64.

    When first_argmax=True, step 0 is argmax(logits) and steps 1..K-1 greedily
    minimize F(S) = sum_j w_j min_{s in S} ||e_s - e_j||_2.
    When first_argmax=False, step 0 is the weighted medoid (matching risk_order).
    Returns (pi [B, K] int64, raw_marginal_gains Delta [B, K] float64).
    """
    end_d = end.double()
    w_d = w.double()
    B, K, _ = end_d.shape
    rows = torch.arange(B, device=end_d.device)
    D = torch.cdist(end_d, end_d)
    tie = torch.arange(K, device=end_d.device, dtype=D.dtype) * 1.0e-12
    taken = torch.zeros(B, K, dtype=torch.bool, device=end_d.device)
    out = torch.empty((B, K), dtype=torch.long, device=end_d.device)
    deltas = torch.zeros((B, K), dtype=D.dtype, device=end_d.device)

    if first_argmax:
        if logits is None:
            raise ValueError("logits required when first_argmax=True")
        c0 = logits.double().argmax(dim=1)
    else:
        obj0 = (w_d[:, None, :] * D).sum(dim=-1) + tie
        c0 = obj0.argmin(dim=1)

    out[:, 0] = c0
    taken[rows, c0] = True
    best = D[rows, c0]
    prev_f = (w_d * best).sum(dim=1)

    for r in range(1, K):
        cand_best = torch.minimum(best[:, None, :], D)
        cand_f = (w_d[:, None, :] * cand_best).sum(dim=-1)
        obj = (cand_f + tie).masked_fill(taken, float("inf"))
        c = obj.argmin(dim=1)
        out[:, r] = c
        taken[rows, c] = True
        new_best = cand_best[rows, c]
        new_f = (w_d * new_best).sum(dim=1)
        deltas[:, r] = (prev_f - new_f).clamp_min(0.0)
        best = new_best
        prev_f = new_f

    return out, deltas


def cover_order_numpy(
    end: np.ndarray,
    w: np.ndarray,
    first_argmax: bool = True,
    logits: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference for cover_order_torch (verified equal in preflight)."""
    end_d = np.asarray(end, dtype=np.float64)
    w_d = np.asarray(w, dtype=np.float64)
    N, K, _ = end_d.shape
    D = np.linalg.norm(end_d[:, :, None] - end_d[:, None], axis=-1)
    tie = np.arange(K, dtype=np.float64) * 1.0e-12
    taken = np.zeros((N, K), dtype=bool)
    out = np.empty((N, K), dtype=np.int64)
    deltas = np.zeros((N, K), dtype=np.float64)
    rows = np.arange(N)

    if first_argmax:
        if logits is None:
            raise ValueError("logits required when first_argmax=True")
        c0 = np.asarray(logits, dtype=np.float64).argmax(axis=1)
    else:
        obj0 = (w_d[:, None, :] * D).sum(axis=-1) + tie
        c0 = obj0.argmin(axis=1)

    out[:, 0] = c0
    taken[rows, c0] = True
    best = D[rows, c0]
    prev_f = (w_d * best).sum(axis=1)

    for r in range(1, K):
        cand_best = np.minimum(best[:, None, :], D)
        cand_f = (w_d[:, None, :] * cand_best).sum(axis=-1)
        obj = cand_f + tie
        obj[taken] = np.inf
        c = obj.argmin(axis=1)
        out[:, r] = c
        taken[rows, c] = True
        new_best = cand_best[rows, c]
        new_f = (w_d * new_best).sum(axis=1)
        deltas[:, r] = np.maximum(prev_f - new_f, 0.0)
        best = new_best
        prev_f = new_f

    return out, deltas


@torch.no_grad()
def order_redundancy_features(
    end: Tensor,
    w: Tensor,
    pi: Tensor,
    sigma_o: float = SIGMA_O,
    first_d: float = FIRST_D_PREV,
) -> dict[str, Tensor]:
    """Compute rank-ordered pairwise distance and slot-aligned d_prev, o, delta."""
    B, K, _ = end.shape
    ep_ord = end.gather(1, pi[..., None].expand_as(end))
    dist_ord = torch.cdist(ep_ord.float(), ep_ord.float())

    triu0 = torch.ones(K, K, device=end.device, dtype=torch.bool).triu(0)
    d_prev_ord = dist_ord.masked_fill(triu0.unsqueeze(0), float("inf")).amin(dim=-1)
    d_prev_ord[:, 0] = float(first_d)
    o_ord = torch.exp(-0.5 * (d_prev_ord / float(sigma_o)).square())
    o_ord[:, 0] = 0.0

    # Marginal coverage reduction along the given ordering pi
    D_ord = torch.cdist(ep_ord.double(), ep_ord.double())
    w_ord = w.double().gather(1, pi)
    deltas_ord = torch.zeros((B, K), dtype=torch.float64, device=end.device)
    best = D_ord[:, 0]
    prev_f = (w_ord * best).sum(dim=1)
    for r in range(1, K):
        new_best = torch.minimum(best, D_ord[:, r])
        new_f = (w_ord * new_best).sum(dim=1)
        deltas_ord[:, r] = (prev_f - new_f).clamp_min(0.0)
        best = new_best
        prev_f = new_f

    denom = deltas_ord[:, 1:].amax(dim=1, keepdim=True).clamp_min(1.0e-12)
    delta_norm_ord = torch.empty((B, K), dtype=torch.float32, device=end.device)
    delta_norm_ord[:, 0] = 1.0
    delta_norm_ord[:, 1:] = (deltas_ord[:, 1:] / denom).float()

    inv_pi = pi.argsort(dim=1)
    return dict(
        dist_ord=dist_ord,
        inv_pi=inv_pi,
        d_prev=d_prev_ord.gather(1, inv_pi),
        o=o_ord.gather(1, inv_pi),
        delta=delta_norm_ord.gather(1, inv_pi),
        rank_norm=inv_pi.float() / max(K - 1, 1),
    )


class WidenedCrossLayer(nn.Module):
    """Capacity-matched Layer 1 without mode self-attention for arm M6."""

    def __init__(self, token_dim: int, width: int = 88, heads: int = 4, ffn_dim: int = 180) -> None:
        super().__init__()
        self.width = int(width)
        self.q_up = nn.Linear(64, self.width)
        self.memory_proj = nn.Linear(token_dim, self.width)
        self.geometric = nn.Linear(6, self.width)
        self.cross_norm = nn.LayerNorm(self.width)
        self.cross = nn.MultiheadAttention(self.width, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(self.width)
        self.ff = nn.Sequential(
            nn.Linear(self.width, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, self.width),
        )
        self.offset = zero(nn.Linear(self.width, 6))
        self.score = nn.Sequential(
            nn.LayerNorm(self.width + 6),
            nn.Linear(self.width + 6, 64),
            nn.GELU(),
            zero(nn.Linear(64, 1)),
        )

    def forward(self, h: Tensor, a_norm_flat: Tensor, mem_raw: Tensor, padding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = self.q_up(h) + self.geometric(a_norm_flat)
        mem = self.memory_proj(mem_raw)
        q = q + self.cross(self.cross_norm(q), mem, mem, key_padding_mask=padding, need_weights=False)[0]
        y = q + self.ff(self.ff_norm(q))
        shift = self.offset(y)
        ds = self.score(torch.cat((y, a_norm_flat), dim=-1))
        return y, shift, ds


class DiscriminativeCoverLayer(nn.Module):
    """Lightweight discriminative ranking layer (no cross-attn, no geometry update).

    Attends causally along order pi over K=20 candidates plus n_s=4 always-visible
    learnable sink tokens, with symmetric pairwise endpoint distance bias.
    """

    def __init__(
        self,
        num_modes: int = 20,
        dim: int = 64,
        heads: int = 4,
        ffn_dim: int = 128,
        num_sinks: int = NUM_SINKS,
        zero_redundancy: bool = False,
    ) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.num_sinks = int(num_sinks)
        self.zero_redundancy = bool(zero_redundancy)
        if self.dim % self.heads != 0:
            raise ValueError("DiscriminativeCoverLayer dim must be divisible by heads")

        # 64 (h) + 6 (a/scale) + 1 (z) + 1 (o) + 1 (log(d_prev+0.05)) + 1 (rank_norm) + 1 (delta) = 75
        self.in_proj = nn.Sequential(nn.LayerNorm(75), nn.Linear(75, self.dim))
        self.rank_emb = nn.Parameter(torch.empty(self.num_modes, self.dim))
        nn.init.normal_(self.rank_emb, std=0.02)
        self.sinks = nn.Parameter(torch.empty(self.num_sinks, self.dim))
        nn.init.normal_(self.sinks, std=0.02)
        self.sink_bias = nn.Parameter(torch.zeros(self.heads, self.num_sinks))

        self.dist_bias = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            zero(nn.Linear(16, self.heads)),
        )

        self.self_norm = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

        self.ff_norm = nn.LayerNorm(self.dim)
        self.ff = nn.Sequential(
            nn.Linear(self.dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, self.dim),
        )
        self.score_head = zero(nn.Linear(self.dim, 1))

    def forward(
        self,
        h: Tensor,
        a_norm_flat: Tensor,
        z: Tensor,
        pi: Tensor,
        red: dict[str, Tensor],
    ) -> Tensor:
        B, K, _ = h.shape
        o = red["o"]
        log_d = torch.log(red["d_prev"] + 0.05)
        delta = red["delta"]
        if self.zero_redundancy:
            o = torch.zeros_like(o)
            log_d = torch.zeros_like(log_d)
            delta = torch.zeros_like(delta)

        feat = torch.cat(
            (
                h,
                a_norm_flat,
                z.unsqueeze(-1),
                o.unsqueeze(-1),
                log_d.unsqueeze(-1),
                red["rank_norm"].unsqueeze(-1),
                delta.unsqueeze(-1),
            ),
            dim=-1,
        )
        inv_pi = red["inv_pi"]
        tok = self.in_proj(feat) + self.rank_emb[inv_pi]
        tok_ord = tok.gather(1, pi[..., None].expand_as(tok))

        q_in = self.self_norm(tok_ord)
        sinks_exp = self.sinks.unsqueeze(0).expand(B, -1, -1)
        kv_in = self.self_norm(torch.cat((sinks_exp, tok_ord), dim=1))

        q = self.q_proj(q_in).reshape(B, K, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_in).reshape(B, self.num_sinks + K, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).reshape(B, self.num_sinks + K, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(float(self.head_dim))

        cand_bias = self.dist_bias(red["dist_ord"].detach().unsqueeze(-1)).permute(0, 3, 1, 2)  # [B, H, K, K]
        s_bias = self.sink_bias.view(1, self.heads, 1, self.num_sinks).expand(B, -1, K, -1)
        attn_logits = attn_logits + torch.cat((s_bias, cand_bias), dim=-1)

        sink_allowed = torch.ones((K, self.num_sinks), dtype=torch.bool, device=h.device)
        cand_allowed = torch.tril(torch.ones((K, K), dtype=torch.bool, device=h.device), diagonal=-1)
        full_allowed = torch.cat((sink_allowed, cand_allowed), dim=-1)
        attn_logits = attn_logits.masked_fill(~full_allowed[None, None], float("-inf"))

        weights = torch.softmax(attn_logits, dim=-1)
        attended = torch.einsum("bhqk,bhkd->bhqd", weights, v).permute(0, 2, 1, 3).reshape(B, K, self.dim)
        u_ord = tok_ord + self.out_proj(attended)
        u_ord = u_ord + self.ff(self.ff_norm(u_ord))

        u = u_ord.gather(1, inv_pi[..., None].expand_as(u_ord))
        dz = 1.5 * torch.tanh(self.score_head(u).squeeze(-1))
        return z + dz


class CascadeCoverDecoder(nn.Module):
    """Layer-differentiated decoder: H3 (L0) + Geometric L1 + optional Discriminative Layer D."""

    def __init__(self, h3: nn.Module, arm: str) -> None:
        super().__init__()
        self.base = h3
        self.arm = arm
        l1_kind, theta0, m0, theta1, m1, disc_order, zero_red, _ = ARM_SPECS[arm]
        self.l1_kind = l1_kind
        self.theta0 = theta0
        self.m0 = m0
        self.theta1 = theta1
        self.m1 = m1
        self.disc_order = disc_order
        self.valid: Optional[Tensor] = None
        self.outs: Optional[list[dict]] = None
        self.disc_out: Optional[dict] = None

        token_dim = (h3.context[0].normalized_shape[0] - 7) // 2
        if l1_kind == "narrow":
            self.memory_proj = nn.Linear(token_dim, 64)
            self.geometric = nn.Linear(6, 64)
            self.layer1 = CrossLayer(self_attention=None)
            self.offset = zero(nn.Linear(64, 6))
            self.score1 = nn.Sequential(
                nn.LayerNorm(70),
                nn.Linear(70, 64),
                nn.GELU(),
                zero(nn.Linear(64, 1)),
            )
            self.widened = None
        elif l1_kind == "widened":
            self.widened = WidenedCrossLayer(token_dim=token_dim, width=88, heads=4, ffn_dim=180)
            self.memory_proj = None
            self.geometric = None
            self.layer1 = None
            self.offset = None
            self.score1 = None
        else:
            raise ValueError(f"Unexpected l1_kind {l1_kind} for {arm}")

        if disc_order is not None:
            self.disc_layer = DiscriminativeCoverLayer(
                num_modes=int(h3.num_modes),
                dim=64,
                heads=4,
                ffn_dim=128,
                num_sinks=NUM_SINKS,
                zero_redundancy=zero_red,
            )
        else:
            self.disc_layer = None

    def raw_memory(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        own, agents = encoded[:, 0], encoded[:, :, -1]
        n = encoded.size(0)
        if self.valid is None:
            raise ValueError("Missing num_valid for decoder cross attention")
        pad = torch.arange(agents.shape[1], device=encoded.device)[None] >= self.valid.reshape(n, 1)
        padding = torch.cat((torch.zeros_like(own[:, :, 0], dtype=torch.bool), pad), 1)
        return torch.cat((own, agents), 1), padding

    def proposal(self, h: Tensor, scale: Tensor) -> Tensor:
        b = self.base
        delta = 0.75 * torch.tanh(b.proposal_delta(h)).reshape(h.size(0), h.size(1), 3, 2)
        return scale[:, None, None] * (b.anchor_templates[None] + delta)

    def trajectory(self, anchors: Tensor, last: Tensor) -> Tensor:
        return last[:, None, None] + torch.einsum("tp,bkpc->bktc", self.base.interpolation, anchors)

    def forward(self, encoded: Tensor, past: Tensor) -> tuple[Tensor, Tensor]:
        b = self.base
        enc, foc = encoded[:, 0], past[:, :1]
        c = b.context(torch.cat((enc[:, -1], enc.mean(1), b.motion_features(foc)[:, 0]), -1))
        q = b.mode_mlp(c[:, None] + b.mode_embedding[None])
        scale = F.softplus(b.proposal_scale(c)) + 0.25
        last = foc[:, 0, -1]

        # Layer 0: H3 proposal
        a0 = self.proposal(q, scale)
        paths0 = self.trajectory(a0, last)
        z0 = b.score_head(torch.cat((q, a0.flatten(2)), -1)).squeeze(-1)
        outs = [dict(paths=paths0, logits=z0, anchors=a0)]

        # Layer 1: Geometric refinement (cross-attn + FFN, no mode self-attn)
        mem_raw, padding = self.raw_memory(encoded)
        a0_norm = (a0 / scale[:, None, None]).flatten(2)
        if self.l1_kind == "narrow":
            x = q + self.geometric(a0_norm)
            y = self.layer1(x, self.memory_proj(mem_raw), padding)
            shift = self.offset(y)
            a1 = a0 + SHIFT * scale[:, None, None] * torch.tanh(shift).reshape(y.size(0), y.size(1), 3, 2)
            a1_norm = (a1 / scale[:, None, None]).flatten(2)
            z1 = z0 + self.score1(torch.cat((y, a1_norm), -1)).squeeze(-1)
        else:
            y, shift, ds = self.widened(q, a0_norm, mem_raw, padding)
            a1 = a0 + SHIFT * scale[:, None, None] * torch.tanh(shift).reshape(y.size(0), y.size(1), 3, 2)
            z1 = z0 + ds.squeeze(-1)

        paths1 = self.trajectory(a1, last)
        outs.append(dict(paths=paths1, logits=z1, anchors=a1))
        self.outs = outs

        if self.disc_layer is None:
            self.disc_out = None
            return paths1[:, None], z1[:, None]

        # Discriminative Layer D (no geometry change, no cross-attn)
        with torch.no_grad():
            e1 = paths1[:, :, -1, :].detach()
            w1 = torch.softmax(z1.detach().double() / TAU_U, dim=-1)
            if self.disc_order == "cover":
                pi, _ = cover_order_torch(e1, w1, first_argmax=True, logits=z1.detach())
            elif self.disc_order == "conf":
                pi = torch.argsort(z1.detach(), dim=1, descending=True, stable=True)
            else:
                raise ValueError(f"Unknown disc_order: {self.disc_order}")
            red = order_redundancy_features(e1, w1, pi)

        a1_norm = (a1 / scale[:, None, None]).flatten(2)
        s = self.disc_layer(y, a1_norm, z1, pi, red)
        self.disc_out = dict(
            paths=paths1,
            logits=s,
            geom_logits=z1.detach(),
            order=pi,
            d_prev=red["d_prev"],
            overlap=red["o"],
            delta=red["delta"],
        )
        return paths1[:, None], s[:, None]


def attach_decoder(model: nn.Module, cfg, arm: str) -> nn.Module:
    """Configure encoder/decoder for M0..M6 or JM4 while preserving paired B0 RNG."""
    if arm not in DECODER_ARMS:
        raise ValueError(f"Unknown decoder arm: {arm}")
    _, _, _, _, _, _, _, use_c2 = ARM_SPECS[arm]
    if use_c2:
        configure_encoder(model, cfg, "E3")
    if arm == "M0":
        return model

    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[]):
        model.decoder = CascadeCoverDecoder(model.decoder.cpu(), arm)
    model.decoder.to(device)
    model.register_forward_pre_hook(set_valid, with_kwargs=True)
    return model


def cascade_layer_loss(
    pred: Tensor,
    target: Tensor,
    logits: Tensor,
    theta: float,
    max_pos: int,
    criterion,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Cascade one-to-many / one-to-one geometric layer loss with per-window normalization."""
    B, _, K, T, _ = pred.shape
    dist = torch.linalg.vector_norm(pred - target.unsqueeze(-3), dim=-1)  # [B, 1, K, T]
    q = dist.mean(-1) + criterion.winner_fde_weight * dist[..., -1]       # [B, 1, K]
    q0 = q[:, 0]

    with torch.no_grad():
        steps = torch.arange(1, T + 1, device=dist.device, dtype=dist.dtype) / T
        match = (dist[:, 0] <= float(theta) * steps).all(dim=-1)  # [B, K]
        q_masked = q0.masked_fill(~match, float("inf"))
        top_vals, top_idx = torch.topk(q_masked, k=int(max_pos), dim=-1, largest=False, sorted=True)
        pos_set = torch.zeros_like(match).scatter(1, top_idx, torch.isfinite(top_vals))
        has_match = match.any(dim=1)
        fallback_oh = F.one_hot(q0.argmin(dim=1), K).bool()
        pos_set = torch.where(has_match[:, None], pos_set, fallback_oh)
        alpha_logits = (-q0 / 0.10).masked_fill(~pos_set, float("-inf"))
        alpha = torch.softmax(alpha_logits, dim=-1)  # [B, K], sums to 1 per window

    # Weighted regression over positive set P^(l)
    pred0 = pred[:, 0]  # [B, K, T, 2]
    target_exp = target[:, 0, None].expand_as(pred0)  # [B, K, T, 2]
    reg_step = criterion.regression.criterion(pred0, target_exp).sum(dim=-1) * criterion.regression.weights
    # ModeQueryLoss does regression(select, target).mean() across (B, T)
    regression = (alpha[..., None] * reg_step).sum(dim=1).mean()

    ep_loss = F.smooth_l1_loss(pred0[:, :, -1, :], target_exp[:, :, -1, :], reduction="none")  # [B, K, 2]
    endpoint = (alpha[..., None] * ep_loss).sum(dim=1).mean()

    s_steps = min(max(criterion.short_horizon_steps, 1), T)
    early_loss = F.smooth_l1_loss(pred0[:, :, :s_steps, :], target_exp[:, :, :s_steps, :], reduction="none")
    early = (alpha[..., None, None] * early_loss).sum(dim=1).mean()

    # Multi-label focal confidence loss
    y = pos_set.to(logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits[:, 0], y, reduction="none")
    focal = ((1.0 - torch.exp(-raw)).square() * raw * (0.25 * y + 0.75 * (1.0 - y))).mean()

    last = pred[..., -1, :]
    distances = torch.cdist(last, last)
    upper = torch.ones_like(distances, dtype=torch.bool).triu(1)
    diversity = F.relu(criterion.diversity_margin - distances[upper]).mean() if upper.any() else pred.new_zeros(())
    cover = (torch.softmax(-q.detach() / 0.10, dim=-1) * q).sum(-1).mean()

    loss = (
        regression
        + criterion.classification_weight * focal
        + criterion.endpoint_weight * endpoint
        + criterion.short_horizon_weight * early
        + criterion.diversity_weight * diversity
        + 0.20 * cover
    )
    return loss, match, pos_set, regression.detach(), focal.detach()


def discriminative_layer_loss(
    pred: Tensor,
    target: Tensor,
    s_logits: Tensor,
    pi: Tensor,
    criterion,
    theta_d: float = THETA_D,
) -> tuple[Tensor, Tensor, Tensor]:
    """One-to-one EMTA focal loss along order pi for the discriminative layer D."""
    B, _, K, T, _ = pred.shape
    dist = torch.linalg.vector_norm(pred - target.unsqueeze(-3), dim=-1)
    with torch.no_grad():
        steps = torch.arange(1, T + 1, device=dist.device, dtype=dist.dtype) / T
        match = (dist[:, 0] <= float(theta_d) * steps).all(dim=-1)  # [B, K]
        sorted_match = match.gather(1, pi)
        first = sorted_match.to(torch.int64).argmax(dim=1)
        fallback = (dist[:, 0].mean(-1) + criterion.winner_fde_weight * dist[:, 0, :, -1]).argmin(dim=1)
        winner = torch.where(match.any(dim=1), pi.gather(1, first[:, None])[:, 0], fallback)
        y = F.one_hot(winner, K).to(s_logits.dtype)

    raw = F.binary_cross_entropy_with_logits(s_logits, y, reduction="none")
    focal_d = ((1.0 - torch.exp(-raw)).square() * raw * (0.25 * y + 0.75 * (1.0 - y))).mean()
    return criterion.classification_weight * focal_d, match, focal_d.detach()


def cascade_decoder_loss(core, model: nn.Module, arm: str, target: Tensor, criterion) -> tuple[Tensor, dict]:
    """Full training loss for M0..M6 and JM4."""
    dec = model.decoder
    _, theta0, m0, theta1, m1, _, _, _ = ARM_SPECS[arm]
    outs = dec.outs
    total = outs[0]["paths"].new_zeros(())
    info = {}

    for idx, (theta, max_pos) in enumerate(((theta0, m0), (theta1, m1))):
        o = outs[idx]
        lv, match, pos_set, reg, cls = cascade_layer_loss(
            o["paths"][:, None], target, o["logits"][:, None], theta, max_pos, criterion
        )
        total = total + lv
        info[f"l{idx}_loss"] = float(lv.detach())
        info[f"l{idx}_reg"] = float(reg)
        info[f"l{idx}_cls"] = float(cls)
        info[f"l{idx}_match_rate"] = float(match.any(1).float().mean())
        info[f"l{idx}_pos_count"] = float(pos_set.sum(1).float().mean())
        if idx == 0:
            info["l0_match_window"] = match.any(1).detach()
            info["l0_pos_window"] = pos_set.sum(1).detach()

    if dec.disc_out is not None:
        d_out = dec.disc_out
        ld, d_match, d_cls = discriminative_layer_loss(
            d_out["paths"][:, None], target, d_out["logits"], d_out["order"], criterion, THETA_D
        )
        total = total + ld
        info["ld_loss"] = float(ld.detach())
        info["ld_cls"] = float(d_cls)
        info["ld_match_rate"] = float(d_match.any(1).float().mean())

    info["regression"] = info["l1_reg"]
    info["classification"] = info.get("ld_cls", info["l1_cls"])
    return total, info
