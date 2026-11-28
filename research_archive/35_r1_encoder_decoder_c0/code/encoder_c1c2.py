"""Encoder C1/C2 implementations and diagnostics on top of the locked B0 (F16+H3) baseline."""
from __future__ import annotations

import math
import statistics
import time
import types
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

ENCODER_ARMS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")


class FactorizedPhysicalTimeBias(nn.Module):
    """Per-head RBF time-resolved bias from distance, closing speed, TTC and DCA.

    With 4 heads x 4 physical channels x 5 RBF centers = 80 zero-initialized parameters.
    At step 0, output is identically 0 so B0 + C2 matches B0 bitwise.
    """

    CHANNELS = ("distance", "closing", "ttc", "dca")

    def __init__(
        self,
        heads: int = 4,
        basis_count: int = 5,
        max_abs_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.basis_count = int(basis_count)
        self.max_abs_bias = float(max_abs_bias)

        self.register_buffer("centers_unsigned", torch.linspace(0.0, 1.0, self.basis_count))
        self.register_buffer("centers_signed", torch.linspace(-1.0, 1.0, self.basis_count))
        self.unsigned_sigma = 0.25
        self.signed_sigma = 0.50

        self.weights = nn.Parameter(
            torch.zeros(self.heads, len(self.CHANNELS), self.basis_count)
        )
        self.last_mean_abs_bias: Optional[Tensor] = None
        self.last_max_abs_bias: Optional[Tensor] = None

    @staticmethod
    def _unsigned_scale(value: Tensor, reference: float = 6.0) -> Tensor:
        denominator = math.log1p(reference)  # log(7)
        return (torch.log1p(value.clamp_min(0.0)) / denominator).clamp(0.0, 1.5)

    def _rbf_unsigned(self, value: Tensor) -> Tensor:
        return torch.exp(
            -0.5 * ((value[..., None] - self.centers_unsigned) / self.unsigned_sigma).square()
        )

    def _rbf_signed(self, value: Tensor) -> Tensor:
        return torch.exp(
            -0.5 * ((value[..., None] - self.centers_signed) / self.signed_sigma).square()
        )

    def forward(self, raw_relation_features: Tensor) -> Tensor:
        """Compute [B, H, N, N, T] bias from raw relation features [B, N, N, T, 8]."""
        if raw_relation_features.ndim != 5 or raw_relation_features.shape[-1] != 8:
            raise ValueError(
                f"Expected raw relation features [B,N,N,T,8], got {tuple(raw_relation_features.shape)}"
            )
        distance = raw_relation_features[..., 4]
        closing = raw_relation_features[..., 5]
        ttc = raw_relation_features[..., 6].clamp(0.0, 1.0)
        dca = raw_relation_features[..., 7]

        distance_n = self._unsigned_scale(distance)
        closing_n = torch.tanh(closing / 1.5)

        approach = torch.sigmoid(closing / 0.20)
        ttc_effective = approach * ttc + (1.0 - approach)
        dca_n = self._unsigned_scale(dca)
        dca_effective = approach * dca_n + (1.0 - approach)

        basis = torch.stack(
            (
                self._rbf_unsigned(distance_n),
                self._rbf_signed(closing_n),
                self._rbf_unsigned(ttc_effective),
                self._rbf_unsigned(dca_effective),
            ),
            dim=-2,
        )
        raw_bias = torch.einsum("bijtcm,hcm->bhijt", basis, self.weights)
        bias = self.max_abs_bias * torch.tanh(raw_bias / max(self.max_abs_bias, 1.0e-6))

        agents = bias.shape[2]
        eye = torch.eye(agents, dtype=torch.bool, device=bias.device).view(1, 1, agents, agents, 1)
        bias = bias.masked_fill(eye, 0.0)

        self.last_mean_abs_bias = bias.detach().abs().mean()
        self.last_max_abs_bias = bias.detach().abs().amax()
        return bias


def lagged_relation_features(encoder: nn.Module, past: Tensor) -> Tensor:
    """Compute C2-lag relation features [B, N, N, T, 8] using receiver i at final step T."""
    # past: [B, N, T, 2]
    frequency = encoder.frequency
    position = past.permute(0, 2, 1, 3)  # [B, T, N, 2]
    velocity = torch.zeros_like(position)
    velocity[:, 1:] = (position[:, 1:] - position[:, :-1]) * frequency
    velocity[:, 0] = velocity[:, 1]

    # receiver i fixed at last observed step T-1; sender j varies over t=0..T-1
    relative_position = position[:, :, None, :, :] - position[:, -1:, :, None, :]
    relative_velocity = velocity[:, :, None, :, :] - velocity[:, -1:, :, None, :]

    if getattr(encoder, "heading_aligned_relations", False):
        receiver_velocity = velocity[:, -1:, :, None, :]
        receiver_speed = torch.linalg.vector_norm(receiver_velocity, dim=-1, keepdim=True)
        default_heading = torch.zeros_like(receiver_velocity)
        default_heading[..., 0] = 1.0
        heading = torch.where(
            receiver_speed > 0.02,
            receiver_velocity / receiver_speed.clamp_min(1.0e-6),
            default_heading,
        )
        cos, sin = heading[..., 0], heading[..., 1]

        def to_local(vectors: Tensor) -> Tensor:
            x, y = vectors[..., 0], vectors[..., 1]
            return torch.stack((x * cos + y * sin, -x * sin + y * cos), dim=-1)

        relative_position = to_local(relative_position)
        relative_velocity = to_local(relative_velocity)

    distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
    dot = (relative_position * relative_velocity).sum(dim=-1, keepdim=True)
    velocity_sq = relative_velocity.square().sum(dim=-1, keepdim=True)
    closing = -dot / distance.clamp_min(1.0e-3)
    ttc = (-dot / velocity_sq.clamp_min(1.0e-4)).clamp(0.0, 10.0)
    closest = relative_position + ttc * relative_velocity
    dca = torch.linalg.vector_norm(closest, dim=-1, keepdim=True)
    return torch.cat(
        (relative_position, relative_velocity, distance, closing, ttc / 10.0, dca),
        dim=-1,
    ).permute(0, 2, 3, 1, 4)


def _select_graph_override(graph_builder, mode: str, distance_last: Tensor, risk_score: Tensor, candidates: Tensor):
    if mode == "radius_full":
        return candidates.clone()
    if mode == "matched_dist_topk":
        # Match E0's adaptive per-row degree exact count, but fill with closest candidates.
        above = (candidates & (risk_score > graph_builder.score_threshold)).sum(dim=-1)
        desired = torch.minimum(
            above.clamp(min=graph_builder.k_min, max=graph_builder.k_max),
            candidates.sum(dim=-1),
        )
        dist_order = distance_last.masked_fill(~candidates, float("inf")).argsort(dim=-1, stable=True)
        rank = torch.arange(distance_last.shape[-1], device=distance_last.device).view(1, 1, -1)
        selected = (rank < desired.unsqueeze(-1)) & candidates.gather(-1, dist_order)
        return torch.zeros_like(candidates).scatter(-1, dist_order, selected)
    raise ValueError(f"Unknown graph override mode: {mode}")


def _install_graph_override(graph_builder, mode: str) -> None:
    """Patch Stage1GraphBuilder build() & build_context() cleanly for E5 / E6."""
    from models.stage1_graph import GraphContext

    graph_builder.c1c2_support_override = mode
    orig_build = graph_builder.build
    orig_build_context = graph_builder.build_context

    def wrapped_build(trajectories, num_valid=None):
        override = getattr(graph_builder, "c1c2_support_override", None)
        if override is None:
            return orig_build(trajectories, num_valid)
        batch_size, _, num_nodes, _ = trajectories.shape
        valid = graph_builder.valid_mask(num_valid, batch_size, num_nodes, trajectories.device)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=trajectories.device).unsqueeze(0)
        non_self = pair_valid & ~eye
        last_position = trajectories[:, -1]
        distance_last = torch.cdist(last_position, last_position)
        candidates = non_self & (distance_last < graph_builder.radius)
        risk_score = graph_builder.kinematic_score(trajectories) if override == "matched_dist_topk" else None
        adjacency = _select_graph_override(graph_builder, override, distance_last, risk_score, candidates)
        if graph_builder.symmetric:
            adjacency = graph_builder._symmetrize(adjacency)
        adjacency = adjacency | (eye & pair_valid)
        graph_builder.last_candidates = candidates.detach()
        graph_builder.last_adjacency = adjacency.detach()
        return adjacency, valid

    def wrapped_build_context(trajectories, num_valid=None):
        ctx = orig_build_context(trajectories, num_valid)
        override = getattr(graph_builder, "c1c2_support_override", None)
        if override is None:
            graph_builder.last_candidates = ctx.candidates.detach()
            graph_builder.last_adjacency = ctx.adjacency.detach()
            return ctx
        batch_size, _, num_nodes, _ = trajectories.shape
        pair_valid = ctx.valid.unsqueeze(2) & ctx.valid.unsqueeze(1)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=trajectories.device).unsqueeze(0)
        non_self = pair_valid & ~eye
        last_dist = torch.cdist(trajectories[:, -1], trajectories[:, -1])
        adjacency = _select_graph_override(
            graph_builder, override, last_dist, ctx.risk_score, ctx.candidates
        )
        if graph_builder.symmetric:
            adjacency = graph_builder._symmetrize(adjacency)
        neighbor_mask = adjacency & non_self
        adjacency = adjacency | (eye & pair_valid)

        # Recompute structural descriptors consistently with the new neighbor_mask
        normalized_risk = graph_builder._masked_risk_distribution(ctx.risk_score, neighbor_mask)
        candidate_count = ctx.candidates.sum(dim=-1).to(trajectories.dtype)
        out_degree = neighbor_mask.sum(dim=-1).to(trajectories.dtype)
        in_degree = neighbor_mask.sum(dim=-2).to(trajectories.dtype)
        degree_imbalance = (out_degree - in_degree) / (out_degree + in_degree + graph_builder.eps)
        risk_total = ctx.risk_score.masked_fill(~neighbor_mask, 0.0).sum(dim=-1)
        risk_max = ctx.risk_score.masked_fill(~neighbor_mask, float("-inf")).max(dim=-1).values
        risk_max = torch.where(torch.isfinite(risk_max), risk_max, torch.zeros_like(risk_max))
        risk_entropy = -(normalized_risk * torch.log(normalized_risk.clamp(min=1e-8))).sum(dim=-1)
        node_features = torch.stack(
            [
                candidate_count / max(num_nodes - 1, 1),
                out_degree / max(num_nodes - 1, 1),
                in_degree / max(num_nodes - 1, 1),
                degree_imbalance,
                risk_total / max(graph_builder.k_max, 1),
                risk_max,
                risk_entropy / max(math.log(float(max(graph_builder.k_max, 2))), graph_builder.eps),
            ],
            dim=-1,
        ) * ctx.valid.unsqueeze(-1)

        adj_float = neighbor_mask.to(trajectories.dtype)
        edge_strength = ctx.risk_score * adj_float
        motif_scale = float(max(graph_builder.k_max, 1))
        fwd_cnt = torch.matmul(adj_float, adj_float) / motif_scale
        bwd_cnt = fwd_cnt.transpose(-1, -2)
        cout_cnt = torch.matmul(adj_float, adj_float.transpose(-1, -2)) / motif_scale
        cin_cnt = torch.matmul(adj_float.transpose(-1, -2), adj_float) / motif_scale
        conv_cnt = in_degree.unsqueeze(-2).expand(-1, num_nodes, -1) / motif_scale
        div_cnt = out_degree.unsqueeze(-1).expand(-1, -1, num_nodes) / motif_scale

        fwd_str = torch.matmul(edge_strength, edge_strength) / motif_scale
        bwd_str = fwd_str.transpose(-1, -2)
        cout_str = torch.matmul(edge_strength, edge_strength.transpose(-1, -2)) / motif_scale
        cin_str = torch.matmul(edge_strength.transpose(-1, -2), edge_strength) / motif_scale
        inc_str = edge_strength.sum(dim=-2) / motif_scale
        out_str = edge_strength.sum(dim=-1) / motif_scale
        conv_str = inc_str.unsqueeze(-2).expand(-1, num_nodes, -1)
        div_str = out_str.unsqueeze(-1).expand(-1, -1, num_nodes)

        edge_structure = torch.stack(
            [
                fwd_cnt, bwd_cnt, cout_cnt, cin_cnt, conv_cnt, div_cnt,
                fwd_str, bwd_str, cout_str, cin_str, conv_str, div_str,
            ],
            dim=-1,
        ) * neighbor_mask.unsqueeze(-1)

        components = graph_builder._history_components(trajectories)
        kinematic_features = graph_builder._kinematic_features(components) * neighbor_mask.unsqueeze(-1)
        graph_builder.last_candidates = ctx.candidates.detach()
        graph_builder.last_adjacency = adjacency.detach()
        return GraphContext(
            adjacency=adjacency,
            valid=ctx.valid,
            candidates=ctx.candidates,
            risk_score=ctx.risk_score,
            risk_history=ctx.risk_history,
            kinematic_features=kinematic_features,
            node_features=node_features,
            edge_structure_features=edge_structure,
        )

    graph_builder.build = wrapped_build
    graph_builder.build_context = wrapped_build_context


def _relation_tokens_c1c2(self, past: Tensor) -> tuple[Optional[Tensor], Tensor, Tensor, Tensor]:
    features, last_distance, last_closing = self.relation_features(past)
    self.last_raw_relation_features = features
    if getattr(self, "c1c2_skip_relation_encoder", False):
        tokens = None
    else:
        encoded = self.relation_encoder(features)
        if self.use_multiscale_relation:
            recent2 = encoded[:, :, :, -2:].mean(dim=3)
            recent4 = encoded[:, :, :, -4:].mean(dim=3)
            full = encoded.mean(dim=3)
            scales = torch.stack((recent2, recent4, full), dim=3)
            scale_weights = torch.softmax(self.relation_scale_score(scales), dim=3)
            tokens = (scale_weights * scales).sum(dim=3)
        else:
            weights = torch.softmax(self.relation_time_score(encoded), dim=3)
            tokens = (weights * encoded).sum(dim=3)
    final_ttc = features[:, :, :, -1, 6] * 10.0
    final_dca = features[:, :, :, -1, 7]
    return tokens, last_distance, last_closing, final_ttc + final_dca


def _temporal_social_attention_c1c2(
    self,
    temporal: Tensor,
    agent: Tensor,
    relation: Optional[Tensor],
    allowed: Tensor,
    type_features: Optional[Tensor] = None,
    risk_score: Optional[Tensor] = None,
) -> Tensor:
    del type_features, risk_score
    batch, agents, steps, _ = temporal.shape
    q = self.q_proj(agent).reshape(batch, agents, self.heads, self.qk_head_dim).permute(0, 2, 1, 3)
    k = self.k_proj(temporal).reshape(
        batch, agents, steps, self.kv_heads, self.qk_head_dim
    ).permute(0, 3, 1, 2, 4)
    v = self.v_proj(temporal).reshape(
        batch, agents, steps, self.kv_heads, self.head_dim
    ).permute(0, 3, 1, 2, 4)
    if self.kv_heads != self.heads:
        repeats = self.heads // self.kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    scores = torch.einsum("bhid,bhjtd->bhijt", q, k) / math.sqrt(float(self.qk_head_dim))

    captured_beta = None
    if self.use_relation_tokens and relation is not None:
        edge_bias = self.relation_bias(relation).permute(0, 3, 1, 2)
        captured_beta = edge_bias
        scores = scores + edge_bias[..., None]

    captured_phys_bias = None
    if getattr(self, "physical_time_bias", None) is not None:
        if getattr(self, "c2_lag", False):
            raw_features = lagged_relation_features(self, self._past)
        else:
            raw_features = self.last_raw_relation_features
        physical_bias = self.physical_time_bias(raw_features)
        captured_phys_bias = physical_bias
        scores = scores + physical_bias

    expanded_mask = allowed[:, None, :, :, None].expand_as(scores)
    scores = scores.masked_fill(~expanded_mask, -1.0e4)
    weights = torch.softmax(scores.flatten(-2), dim=-1).reshape_as(scores)

    base = v[:, :, None].expand(-1, -1, agents, -1, -1, -1)
    captured_edge_val = None
    if self.use_relation_tokens and relation is not None:
        edge_value = self.relation_value(relation).reshape(
            batch, agents, agents, self.heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        captured_edge_val = edge_value
        message = base + edge_value[..., None, :]
    else:
        message = base

    if getattr(self, "capture_attention_diagnostics", False):
        self.last_attn_weights = weights.detach()
        self.last_attn_allowed = allowed.detach()
        self.last_beta = (
            captured_beta.detach()
            if captured_beta is not None
            else torch.zeros(batch, self.heads, agents, agents, device=scores.device, dtype=scores.dtype)
        )
        self.last_edge_value = (
            captured_edge_val.detach()
            if captured_edge_val is not None
            else torch.zeros(batch, self.heads, agents, agents, self.head_dim, device=scores.device, dtype=scores.dtype)
        )
        self.last_phys_bias = (
            captured_phys_bias.detach()
            if captured_phys_bias is not None
            else torch.zeros_like(scores.detach())
        )

    social = (weights[..., None] * message).sum(dim=(3, 4))
    social = social.permute(0, 2, 1, 3).reshape(batch, agents, self.dim)
    return self.out_proj(social)


def _social_attention_diag(
    self,
    agent: Tensor,
    relation: Optional[Tensor],
    allowed: Tensor,
    risk_score: Optional[Tensor] = None,
) -> Tensor:
    del risk_score
    batch, agents, _ = agent.shape
    q = self.q_proj(agent).reshape(batch, agents, self.heads, self.head_dim).permute(0, 2, 1, 3)
    k = self.k_proj(agent).reshape(batch, agents, self.heads, self.head_dim).permute(0, 2, 1, 3)
    v = self.v_proj(agent).reshape(batch, agents, self.heads, self.head_dim).permute(0, 2, 1, 3)
    scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(float(self.head_dim))
    if self.use_relation_tokens and relation is not None:
        scores = scores + self.relation_bias(relation).permute(0, 3, 1, 2)
    attention_mask = allowed if allowed.ndim == 4 else allowed[:, None]
    scores = scores.masked_fill(~attention_mask, -1.0e4)
    weights = torch.softmax(scores, dim=-1)
    base_message = v[:, :, None].expand(-1, -1, agents, -1, -1)
    if self.use_relation_tokens and relation is not None:
        edge_value = self.relation_value(relation).reshape(
            batch, agents, agents, self.heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        message = base_message + edge_value
    else:
        message = base_message
    if getattr(self, "capture_attention_diagnostics", False):
        self.last_attn_weights = weights.detach()
        self.last_attn_allowed = allowed.detach()
    social = (weights[..., None] * message).sum(dim=3)
    social = social.permute(0, 2, 1, 3).reshape(batch, agents, self.dim)
    return self.out_proj(social)


def _validate_locked_f16(base: nn.Module) -> None:
    required_false = (
        "use_pair_deformable_temporal",
        "use_pair_temporal_soft_prior",
        "use_scaled_cosine_attention",
        "use_differential_attention",
        "use_relation_type_bias",
        "use_soft_risk_bias",
        "use_temporal_relative_bias",
        "use_sparsemax",
        "use_entmax15",
        "use_relation_grouped_vector_attention",
        "use_temporal_half_pool",
        "use_fixed_half_projection",
        "use_recent_exact_early_pool",
        "use_motion_event_sampling",
        "use_motion_token_squeeze",
        "use_edge_list_attention",
    )
    active = [name for name in required_false if bool(getattr(base, name, False))]
    if active:
        raise ValueError(f"Unexpected active flags on locked F16 encoder: {active}")
    if not bool(getattr(base, "use_temporal_social_cross", False)):
        raise ValueError("Locked F16 encoder requires use_temporal_social_cross=True")
    if not bool(getattr(base, "use_relation_tokens", False)):
        raise ValueError("Locked F16 encoder requires use_relation_tokens=True")


def configure_encoder(model: nn.Module, cfg, arm: str, seed: int = 42) -> nn.Module:
    """Apply C1/C2 encoder intervention arm E0..E6 on a freshly built B0 model."""
    del cfg, seed
    if arm not in ENCODER_ARMS:
        raise ValueError(f"Unknown encoder arm: {arm}")
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError(f"Expected f16_temporal_social_cross, got {getattr(model, 'front_variant', None)}")

    base = model.agent_aware_trans_encoder
    _validate_locked_f16(base)

    base.c1c2_arm = arm
    base.c1c2_skip_relation_encoder = False
    base.physical_time_bias = None
    base.c2_lag = False
    base.last_raw_relation_features = None
    base.capture_attention_diagnostics = False

    # Always install unified diagnostic-capable relation_tokens & attention wrappers.
    base.relation_tokens = types.MethodType(_relation_tokens_c1c2, base)
    base.temporal_social_attention = types.MethodType(_temporal_social_attention_c1c2, base)
    base.social_attention = types.MethodType(_social_attention_diag, base)

    if arm == "E0":
        return model

    if arm == "E1":
        # Remove relation tokens (beta = 0 and W_e * e = 0) while preserving state_dict keys/shapes.
        base.use_relation_tokens = False
        base.c1c2_skip_relation_encoder = True
        for name in (
            "relation_encoder",
            "relation_time_score",
            "relation_scale_score",
            "relation_bias",
            "relation_value",
        ):
            mod = getattr(base, name)
            for p in mod.parameters():
                p.requires_grad_(False)
        return model

    if arm == "E2":
        # Exact f4_relation_tokens behavior (summary social attention, single-scale relation).
        base.use_temporal_social_cross = False
        base.use_multiscale_relation = False
        return model

    if arm in ("E3", "E4"):
        device = next(base.parameters()).device
        with torch.random.fork_rng(devices=[]):
            ptb = FactorizedPhysicalTimeBias(heads=int(base.heads), basis_count=5, max_abs_bias=2.0).to(device)
        base.physical_time_bias = ptb
        base.c2_lag = arm == "E4"
        return model

    if arm == "E5":
        _install_graph_override(model.graph_builder, "radius_full")
        return model

    if arm == "E6":
        _install_graph_override(model.graph_builder, "matched_dist_topk")
        return model

    raise ValueError(arm)


def compute_support_stats(trajectories_btn2: Tensor, num_valid: Tensor, graph_builder) -> dict:
    """Compute V0-a support-set width and candidate-count distributions on [B, T, N, 2]."""
    with torch.no_grad():
        adj, valid = graph_builder.build(trajectories_btn2, num_valid)
        B, _, N, _ = trajectories_btn2.shape
        eye = torch.eye(N, dtype=torch.bool, device=trajectories_btn2.device).unsqueeze(0)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye
        dist_last = torch.cdist(trajectories_btn2[:, -1], trajectories_btn2[:, -1])
        candidates = pair_valid & (dist_last < graph_builder.radius)
        cand_counts = candidates.sum(dim=-1)[valid].cpu().numpy()
        non_self_widths = (adj & ~eye)[valid].sum(dim=-1).cpu().numpy()
        total_widths = adj[valid].sum(dim=-1).cpu().numpy()
    return dict(
        valid_agent_rows=int(len(non_self_widths)),
        radius_m=float(graph_builder.radius),
        k_min=int(graph_builder.k_min),
        k_max=int(graph_builder.k_max),
        nonself_width_mean=float(np.mean(non_self_widths)),
        nonself_width_p50=float(np.quantile(non_self_widths, 0.50)),
        nonself_width_p90=float(np.quantile(non_self_widths, 0.90)),
        nonself_width_max=int(np.max(non_self_widths)),
        total_width_mean=float(np.mean(total_widths)),
        total_width_max=int(np.max(total_widths)),
        frac_at_kmax_6=float(np.mean(non_self_widths == graph_builder.k_max)),
        frac_candidates_gt_kmax=float(np.mean(cand_counts > graph_builder.k_max)),
        cand_count_mean=float(np.mean(cand_counts)),
        cand_count_p50=float(np.quantile(cand_counts, 0.50)),
        cand_count_p90=float(np.quantile(cand_counts, 0.90)),
        cand_count_p95=float(np.quantile(cand_counts, 0.95)),
        cand_count_max=int(np.max(cand_counts)),
        cand_count_bins={
            "le_3": float(np.mean(cand_counts <= 3)),
            "4_to_6": float(np.mean((cand_counts >= 4) & (cand_counts <= 6))),
            "7_to_12": float(np.mean((cand_counts >= 7) & (cand_counts <= 12))),
            "gt_12": float(np.mean(cand_counts > 12)),
        },
        nonself_width_hist={str(k): int((non_self_widths == k).sum()) for k in range(int(non_self_widths.max()) + 1)},
    )


@torch.no_grad()
def encoder_diagnostics(core, model: nn.Module, data: dict, arm: str, device: str,
                        e0_attn_argmax: Optional[dict] = None) -> tuple[dict, Optional[dict]]:
    """Run deterministic diagnostics over full dataset for MC-1, MC-2, MC-3 and attention mass."""
    model.eval()
    enc = model.agent_aware_trans_encoder
    enc.capture_attention_diagnostics = True

    beta_abs_sum = 0.0
    beta_abs_count = 0
    edge_val_abs_sum = 0.0
    edge_val_abs_count = 0

    b_abs_sum = 0.0
    b_abs_count = 0
    b_rel_std_sum = 0.0
    b_rel_std_count = 0

    e5_mismatch = 0
    e6_mismatch = 0
    total_valid_rows = 0

    joint_argmax_chunks = []
    marg_t_argmax_chunks = []
    pair_t_argmax_chunks = []
    valid_row_masks = []
    valid_pair_masks = []

    try:
        for start in range(0, len(data["ids"]), 64):
            sl = slice(start, start + 64)
            p = torch.as_tensor(data["past"][sl], device=device)
            n = torch.as_tensor(data["num_valid"][sl], device=device)
            _, _, graph, *_ = core.prepare(p)
            core.prediction(model, p, n)

            B, N, _, _ = graph.shape
            valid_nodes = torch.arange(N, device=device)[None] < n[:, None]
            eye = torch.eye(N, dtype=torch.bool, device=device)[None]

            if enc.use_temporal_social_cross:
                allowed = enc.last_attn_allowed  # [B, N, N]
                non_self_allowed = allowed & ~eye & valid_nodes[:, :, None] & valid_nodes[:, None, :]

                beta = enc.last_beta  # [B, H, N, N]
                edge_val = enc.last_edge_value  # [B, H, N, N, d_h]
                m_pair = non_self_allowed[:, None].expand_as(beta)
                if m_pair.any():
                    beta_vals = beta[m_pair].abs()
                    beta_abs_sum += float(beta_vals.sum())
                    beta_abs_count += int(beta_vals.numel())
                    ev_vals = edge_val[m_pair].abs()
                    edge_val_abs_sum += float(ev_vals.sum())
                    edge_val_abs_count += int(ev_vals.numel())

                phys_b = enc.last_phys_bias  # [B, H, N, N, T]
                m_time = non_self_allowed[:, None, :, :, None].expand_as(phys_b)
                if m_time.any():
                    b_vals = phys_b[m_time].abs()
                    b_abs_sum += float(b_vals.sum())
                    b_abs_count += int(b_vals.numel())

                    # Per-pair temporal variation: std_t(b) / (mean_t|b| + 1e-6)
                    b_sel = phys_b[m_pair]  # [P, T]
                    std_t = b_sel.std(dim=-1, unbiased=False)
                    mean_abs_t = b_sel.abs().mean(dim=-1)
                    rel_std = std_t / (mean_abs_t + 1.0e-6)
                    b_rel_std_sum += float(rel_std.sum())
                    b_rel_std_count += int(rel_std.numel())

                w = enc.last_attn_weights  # [B, H, N, N, T]
                # Mask out unallowed slots cleanly before argmax
                w_masked = w.masked_fill(~allowed[:, None, :, :, None], -1.0)
                joint_idx = w_masked.flatten(-2).argmax(dim=-1)  # [B, H, N]
                marg_t_idx = w.sum(dim=3).argmax(dim=-1)  # [B, H, N]
                pair_t_idx = w.argmax(dim=-1)  # [B, H, N, N]
                joint_argmax_chunks.append(joint_idx.cpu())
                marg_t_argmax_chunks.append(marg_t_idx.cpu())
                pair_t_argmax_chunks.append(pair_t_idx.cpu())
                valid_row_masks.append(valid_nodes[:, None, :].expand(-1, enc.heads, -1).cpu())
                valid_pair_masks.append(non_self_allowed[:, None, :, :].expand(-1, enc.heads, -1, -1).cpu())

            # MC-3 checks on graph_builder
            gb = model.graph_builder
            ctx = gb.build_context(graph.transpose(1, 2), n)
            row_width = ctx.adjacency[valid_nodes].sum(dim=-1)
            cand_count = ctx.candidates[valid_nodes].sum(dim=-1)
            total_valid_rows += int(row_width.numel())
            if arm == "E5":
                e5_mismatch += int((row_width != cand_count + 1).sum())
            elif arm == "E6":
                # Compare width against E0's exact adaptive width
                above = (ctx.candidates & (ctx.risk_score > gb.score_threshold)).sum(dim=-1)
                e0_width = torch.minimum(above.clamp(min=gb.k_min, max=gb.k_max), ctx.candidates.sum(dim=-1))[valid_nodes] + 1
                e6_mismatch += int((row_width != e0_width).sum())
    finally:
        enc.capture_attention_diagnostics = False

    diag = dict(
        arm=arm,
        total_valid_rows=total_valid_rows,
        beta_abs_mean=beta_abs_sum / max(beta_abs_count, 1),
        edge_value_abs_mean=edge_val_abs_sum / max(edge_val_abs_count, 1),
        phys_bias_abs_mean=b_abs_sum / max(b_abs_count, 1),
        phys_bias_rel_time_std_mean=b_rel_std_sum / max(b_rel_std_count, 1),
        e5_width_mismatch_rows=e5_mismatch,
        e6_width_mismatch_rows=e6_mismatch,
    )

    attn_bundle = None
    if joint_argmax_chunks:
        attn_bundle = dict(
            joint=torch.cat(joint_argmax_chunks, dim=0),
            marg_t=torch.cat(marg_t_argmax_chunks, dim=0),
            pair_t=torch.cat(pair_t_argmax_chunks, dim=0),
            row_mask=torch.cat(valid_row_masks, dim=0),
            pair_mask=torch.cat(valid_pair_masks, dim=0),
        )
        if e0_attn_argmax is not None:
            rm = attn_bundle["row_mask"]
            pm = attn_bundle["pair_mask"]
            diag["joint_jt_argmax_changed_rate"] = (
                float((attn_bundle["joint"][rm] != e0_attn_argmax["joint"][rm]).float().mean())
                if rm.any() else 0.0
            )
            diag["marginal_t_argmax_changed_rate"] = (
                float((attn_bundle["marg_t"][rm] != e0_attn_argmax["marg_t"][rm]).float().mean())
                if rm.any() else 0.0
            )
            diag["pair_t_argmax_changed_rate"] = (
                float((attn_bundle["pair_t"][pm] != e0_attn_argmax["pair_t"][pm]).float().mean())
                if pm.any() else 0.0
            )

    if arm == "E1":
        diag["mc1_pass"] = diag["beta_abs_mean"] == 0.0 and diag["edge_value_abs_mean"] == 0.0
    if arm in ("E3", "E4"):
        jt_chg = diag.get("joint_jt_argmax_changed_rate")
        pt_chg = diag.get("pair_t_argmax_changed_rate")
        mt_chg = diag.get("marginal_t_argmax_changed_rate")
        diag["mc2_pass"] = bool(
            diag["phys_bias_abs_mean"] >= 0.05
            and diag["phys_bias_rel_time_std_mean"] >= 0.20
            and (
                (jt_chg is not None and jt_chg >= 0.10)
                or (pt_chg is not None and pt_chg >= 0.10)
                or (mt_chg is not None and mt_chg >= 0.10)
            )
        )
    if arm == "E5":
        diag["mc3_pass"] = e5_mismatch == 0
    if arm == "E6":
        diag["mc3_pass"] = e6_mismatch == 0

    return diag, attn_bundle


def run_v2_scaling(
    device: str = "cuda:0",
    n_values: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    batch_sizes: tuple[int, ...] = (1, 64),
    warmup: int = 5,
    iterations: int = 20,
    cycles: int = 4,
) -> list[dict]:
    """V2 efficiency scaling across N and batch size for dense E0, edge-list, fixed-K, and radius-full E5."""
    from front_encoder_models import FactorizedStateRelationEncoder
    from sparse_exec_adapter import edge_relation_features, edge_relation_tokens, _edge_attention
    from fixed_k_adapter import allowed_to_slots, fixed_relation_features, fixed_relation_tokens, _fixed_attention

    torch_dev = torch.device(device)
    rows: list[dict] = []
    obs_len, dim, heads, ffn_dim = 8, 96, 4, 192

    def make_compact_forward(enc: nn.Module, mode: str):
        # Avoid allocating [B*H, 8N, 8N] mask matrices when benchmarking large N (128, 256)
        def compact_dense(src: Tensor, allowed: Tensor) -> Tensor:
            batch, _, _ = src.shape
            tokens = src.reshape(batch, enc.obs_len, enc.agents, dim).permute(0, 2, 1, 3).contiguous()
            temporal = enc.temporal_encoder(tokens.reshape(batch * enc.agents, enc.obs_len, dim)).reshape(
                batch, enc.agents, enc.obs_len, dim
            )
            temporal, agent = enc.temporal_queries(temporal)
            relation, _, _, _ = enc.relation_tokens(enc._past)
            social = enc.temporal_social_attention(temporal, agent, relation, allowed)
            gate = torch.sigmoid(enc.social_gate(torch.cat((agent, social), dim=-1)))
            fused = enc.social_norm(agent + gate * social)
            enriched = enc.output_norm(temporal + enc.token_feedback(fused)[:, :, None, :])
            return enriched.permute(0, 2, 1, 3).reshape(batch, enc.obs_len * enc.agents, dim)

        def compact_edge(src: Tensor, allowed: Tensor) -> Tensor:
            batch, _, _ = src.shape
            tokens = src.reshape(batch, enc.obs_len, enc.agents, dim).permute(0, 2, 1, 3).contiguous()
            temporal = enc.temporal_encoder(tokens.reshape(batch * enc.agents, enc.obs_len, dim)).reshape(
                batch, enc.agents, enc.obs_len, dim
            )
            temporal, agent = enc.temporal_queries(temporal)
            edges = allowed.nonzero(as_tuple=False)
            features = edge_relation_features(enc, enc._past, edges)
            relation, _ = edge_relation_tokens(enc, features)
            social = _edge_attention(enc, temporal, agent, edges, relation, allowed)
            gate = torch.sigmoid(enc.social_gate(torch.cat((agent, social), dim=-1)))
            fused = enc.social_norm(agent + gate * social)
            enriched = enc.output_norm(temporal + enc.token_feedback(fused)[:, :, None, :])
            return enriched.permute(0, 2, 1, 3).reshape(batch, enc.obs_len * enc.agents, dim)

        def compact_fixed(src: Tensor, allowed: Tensor) -> Tensor:
            batch, _, _ = src.shape
            tokens = src.reshape(batch, enc.obs_len, enc.agents, dim).permute(0, 2, 1, 3).contiguous()
            temporal = enc.temporal_encoder(tokens.reshape(batch * enc.agents, enc.obs_len, dim)).reshape(
                batch, enc.agents, enc.obs_len, dim
            )
            temporal, agent = enc.temporal_queries(temporal)
            index, slot_valid = allowed_to_slots(allowed, 6)
            features = fixed_relation_features(enc, enc._past, index)
            relation, _ = fixed_relation_tokens(enc, features)
            social = _fixed_attention(enc, temporal, agent, relation, index, slot_valid, allowed)
            gate = torch.sigmoid(enc.social_gate(torch.cat((agent, social), dim=-1)))
            fused = enc.social_norm(agent + gate * social)
            enriched = enc.output_norm(temporal + enc.token_feedback(fused)[:, :, None, :])
            return enriched.permute(0, 2, 1, 3).reshape(batch, enc.obs_len * enc.agents, dim)

        return {"dense": compact_dense, "edge": compact_edge, "fixed": compact_fixed}[mode]

    for batch_size in batch_sizes:
        for agents in n_values:
            kwargs = dict(
                obs_len=obs_len, agents=agents, dim=dim, heads=heads, ffn_dim=ffn_dim,
                dropout=0.0, use_relation_tokens=True, use_dual_graph=False, top_k=6,
                frequency=2.5, use_multiscale_relation=True, use_temporal_social_cross=True,
            )
            base_enc = FactorizedStateRelationEncoder(**kwargs).to(torch_dev).eval()
            # Build synthetic scene on a compact 2D grid (1.5m spacing -> realistic 6m neighborhood)
            side = math.ceil(math.sqrt(agents))
            gx = (torch.arange(agents, device=torch_dev) % side).float() * 1.5
            gy = (torch.arange(agents, device=torch_dev) // side).float() * 1.5
            pos0 = torch.stack((gx, gy), dim=-1)  # [N, 2]
            t_grid = torch.arange(obs_len, device=torch_dev).float()[None, :, None] * 0.2
            past = pos0[None, :, None, :] + t_grid * torch.tensor([0.5, 0.1], device=torch_dev)
            past = past.expand(batch_size, -1, -1, -1).contiguous()
            src = torch.randn(batch_size, obs_len * agents, dim, device=torch_dev)
            counts = torch.full((batch_size,), agents, device=torch_dev)

            # Sparse E0 allowed (self + up to 6 nearest within 6m) vs E5 radius-full (<6m)
            dist = torch.cdist(past[0, :, -1], past[0, :, -1])  # [N, N]
            eye = torch.eye(agents, dtype=torch.bool, device=torch_dev)
            cand = (~eye) & (dist < 6.0)
            k_cap = min(6, max(agents - 1, 0))
            allowed_e5 = (cand | eye)[None].expand(batch_size, -1, -1).contiguous()
            if k_cap > 0:
                order = dist.masked_fill(~cand, float("inf")).argsort(dim=-1, stable=True)[:, :k_cap]
                sel = torch.zeros_like(cand).scatter(-1, order, cand.gather(-1, order))
                allowed_e0 = (sel | eye)[None].expand(batch_size, -1, -1).contiguous()
            else:
                allowed_e0 = eye[None].expand(batch_size, -1, -1).contiguous()

            sparse_edges = int(allowed_e0[0].sum().item())
            e5_edges = int(allowed_e5[0].sum().item())
            dense_pairs = agents * agents
            fixed_slots = min(agents, 7)

            impls = [
                ("masked_dense_E0", make_compact_forward(base_enc, "dense"), allowed_e0, batch_size * dense_pairs * obs_len * heads),
                ("edge_list_sparse", make_compact_forward(base_enc, "edge"), allowed_e0, batch_size * sparse_edges * obs_len * heads),
                ("fixed_k_sparse", make_compact_forward(base_enc, "fixed"), allowed_e0, batch_size * agents * fixed_slots * obs_len * heads),
                ("radius_full_E5", make_compact_forward(base_enc, "dense"), allowed_e5, batch_size * e5_edges * obs_len * heads),
            ]
            # Wait: E5 uses dense execution, so its element count in dense tensor ops is B * N * N * T * H
            impls[-1] = ("radius_full_E5", impls[-1][1], allowed_e5, batch_size * dense_pairs * obs_len * heads)

            base_enc.set_context(past, counts)
            for name, fn, mask_tensor, attn_elems in impls:
                row = dict(
                    batch_size=batch_size,
                    n=agents,
                    impl=name,
                    sparse_edges_per_scene=sparse_edges,
                    e5_edges_per_scene=e5_edges,
                    dense_pairs_per_scene=dense_pairs,
                    attention_elements=int(attn_elems),
                    status="OK",
                )
                try:
                    with torch.inference_mode():
                        for _ in range(warmup):
                            fn(src, mask_tensor)
                        if torch_dev.type == "cuda":
                            torch.cuda.synchronize()
                        times = []
                        for _ in range(cycles):
                            t0 = time.perf_counter()
                            for _ in range(iterations):
                                fn(src, mask_tensor)
                            if torch_dev.type == "cuda":
                                torch.cuda.synchronize()
                            times.append(1000.0 * (time.perf_counter() - t0) / iterations)
                        row["mean_ms"] = float(statistics.fmean(times))
                        row["median_ms"] = float(statistics.median(times))
                        row["p95_ms"] = float(np.quantile(times, 0.95))
                        if torch_dev.type == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats(torch_dev)
                            before_a = torch.cuda.memory_allocated(torch_dev)
                            before_r = torch.cuda.memory_reserved(torch_dev)
                            fn(src, mask_tensor)
                            torch.cuda.synchronize()
                            row["peak_allocated_mb"] = float(max(0, torch.cuda.max_memory_allocated(torch_dev) - before_a) / (1024 ** 2))
                            row["peak_reserved_mb"] = float(max(0, torch.cuda.max_memory_reserved(torch_dev) - before_r) / (1024 ** 2))
                        else:
                            row["peak_allocated_mb"] = 0.0
                            row["peak_reserved_mb"] = 0.0
                except torch.cuda.OutOfMemoryError:
                    if torch_dev.type == "cuda":
                        torch.cuda.empty_cache()
                    row["status"] = "OOM"
                    row["mean_ms"] = None
                    row["median_ms"] = None
                    row["p95_ms"] = None
                    row["peak_allocated_mb"] = None
                    row["peak_reserved_mb"] = None
                rows.append(row)
            del base_enc
            if torch_dev.type == "cuda":
                torch.cuda.empty_cache()
    return rows
