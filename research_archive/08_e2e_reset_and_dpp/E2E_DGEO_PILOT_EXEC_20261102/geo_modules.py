import math
import torch
from torch import nn
from torch.nn import functional as F
from hierarchical_decoder_models import HierarchicalLearnedAnchorDecoder


def compute_anchor_geo_features_12d(relative_anchors: torch.Tensor, past_loc_focal: torch.Tensor) -> torch.Tensor:
    a1 = relative_anchors[:, :, 0, :]
    a2 = relative_anchors[:, :, 1, :]
    a3 = relative_anchors[:, :, 2, :]
    u1 = a1 / 4.0
    u3 = (a3 - a2) / 4.0
    v_obs = (past_loc_focal[:, -1, :] - past_loc_focal[:, -2, :]).unsqueeze(1)
    junc_dv = u1 - v_obs
    turn_dv = u3 - u1
    end_dist = torch.linalg.vector_norm(a3, dim=-1, keepdim=True)
    end_angle = (torch.atan2(a3[..., 1], a3[..., 0]) / math.pi).unsqueeze(-1)
    return torch.cat([a1, a2, a3, junc_dv, turn_dv, end_dist, end_angle], dim=-1)


class GeoModeCoordinator(nn.Module):
    def __init__(self, dim: int = 64, proj_dim: int = 16, heads: int = 4,
                 use_geo: bool = True, differential: bool = False,
                 lambda_fixed: float = 0.5, out_gain: float = 0.1) -> None:
        super().__init__()
        if proj_dim % heads != 0:
            raise ValueError("HEAD_DIMENSION")
        self.dim = dim
        self.proj_dim = proj_dim
        self.heads = heads
        self.head_dim = proj_dim // heads
        self.use_geo = use_geo
        self.differential = differential
        self.lambda_fixed = float(lambda_fixed)

        if use_geo:
            self.geo_proj = nn.Linear(12, dim, bias=True)
        self.qk_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        self.q1 = nn.Linear(dim, proj_dim, bias=False)
        self.k1 = nn.Linear(dim, proj_dim, bias=False)
        self.value = nn.Linear(dim, proj_dim, bias=False)
        if differential:
            self.q2 = nn.Linear(dim, proj_dim, bias=False)
            self.k2 = nn.Linear(dim, proj_dim, bias=False)
        self.gate = nn.Linear(dim * 2, proj_dim, bias=True)
        self.onorm = nn.LayerNorm(proj_dim)
        self.out_delta = nn.Linear(proj_dim, dim, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.out_delta.weight, gain=out_gain)
        self.diagnostics = None

    def forward(self, z_base: torch.Tensor, context_focal: torch.Tensor,
                geo_feat_12d: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_geo:
            if geo_feat_12d is None:
                raise ValueError("MISSING_GEO_FEATURES")
            z = z_base + self.geo_proj(geo_feat_12d)
            qk_in = self.qk_norm(z - z.mean(dim=1, keepdim=True))
        else:
            z = z_base
            qk_in = self.qk_norm(z)
        v_in = self.v_norm(z)

        b, k, _ = z.shape
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(b, k, self.heads, self.head_dim).transpose(1, 2)

        q1, k1 = split_heads(self.q1(qk_in)), split_heads(self.k1(qk_in))
        v = split_heads(self.value(v_in))
        a1 = torch.softmax((q1 @ k1.transpose(-1, -2)) / math.sqrt(self.head_dim), dim=-1)
        weights = a1
        if self.differential:
            q2, k2 = split_heads(self.q2(qk_in)), split_heads(self.k2(qk_in))
            a2 = torch.softmax((q2 @ k2.transpose(-1, -2)) / math.sqrt(self.head_dim), dim=-1)
            weights = a1 - self.lambda_fixed * a2

        agg = (weights @ v).transpose(1, 2).reshape(b, k, self.proj_dim)
        ctx_exp = context_focal.unsqueeze(1).expand(-1, k, -1)
        gate_val = torch.sigmoid(self.gate(torch.cat([qk_in, ctx_exp], dim=-1)))
        update = self.out_delta(gate_val * self.onorm(agg))

        with torch.no_grad():
            qn = F.normalize(qk_in, dim=-1)
            cos_mat = qn @ qn.transpose(-1, -2)
            off_diag = cos_mat[:, ~torch.eye(k, dtype=torch.bool, device=z.device)].mean()
            self.diagnostics = {
                "negative_fraction": (weights < 0).float().mean().detach(),
                "off_diag_cos_mean": off_diag.detach(),
                "gate_mean": gate_val.mean().detach(),
                "update_norm": torch.linalg.vector_norm(update, dim=-1).detach(),
                "attention_weights": weights.detach(),
            }
        return z_base + update


class PostProposalRelationalScorer(nn.Module):
    def __init__(self, mode_dim: int = 64, geo_dim: int = 12, hidden: int = 16) -> None:
        super().__init__()
        in_dim = mode_dim + geo_dim
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden, bias=True)
        self.out_logit = nn.Linear(hidden * 2 + 2, 1, bias=False)

        nn.init.xavier_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.out_logit.weight)

    def forward(self, mode_focal: torch.Tensor, post_anchors: torch.Tensor,
                past_loc_focal: torch.Tensor) -> torch.Tensor:
        geo_post = compute_anchor_geo_features_12d(post_anchors, past_loc_focal)
        s = F.gelu(self.proj(self.norm(torch.cat([mode_focal, geo_post], dim=-1))))
        s_rel = s - s.mean(dim=1, keepdim=True)
        end_pts = post_anchors[:, :, -1, :]
        dist_sq = torch.cdist(end_pts, end_pts).square()
        density = torch.exp(-dist_sq / 0.25).mean(dim=-1, keepdim=True)
        junc_norm = torch.linalg.vector_norm(geo_post[:, :, 6:8], dim=-1, keepdim=True)
        return self.out_logit(torch.cat([s, s_rel, density, junc_norm], dim=-1)).squeeze(-1)


class DGeoDecoder(HierarchicalLearnedAnchorDecoder):
    def __init__(self, base: HierarchicalLearnedAnchorDecoder, arm: str) -> None:
        nn.Module.__init__(self)
        if base.use_refiner:
            raise ValueError("ONLY_ORIGINAL_H3_NO_REFINER")
        for name, child in base._modules.items():
            self.add_module(name, child)
        for name, param in base._parameters.items():
            self.register_parameter(name, param)
        for name, buf in base._buffers.items():
            self.register_buffer(name, buf)
        for name in ("num_modes", "pred_len", "anchor_steps", "anchor_count", "use_refiner", "top_m"):
            setattr(self, name, getattr(base, name))

        self.arm = arm
        dim = base.mode_embedding.shape[-1]
        if arm == "DG-Base":
            self.coordinator = GeoModeCoordinator(dim=dim, proj_dim=16, heads=4, use_geo=False, differential=False)
            self.rel_scorer = None
        elif arm == "DG-GeoGate":
            self.coordinator = GeoModeCoordinator(dim=dim, proj_dim=16, heads=4, use_geo=True, differential=False)
            self.rel_scorer = None
        elif arm == "DG-GeoDiff":
            self.coordinator = GeoModeCoordinator(dim=dim, proj_dim=16, heads=4, use_geo=True, differential=True)
            self.rel_scorer = None
        elif arm == "DG-GeoScore":
            self.coordinator = GeoModeCoordinator(dim=dim, proj_dim=16, heads=4, use_geo=True, differential=False)
            self.rel_scorer = PostProposalRelationalScorer(mode_dim=dim, geo_dim=12, hidden=16)
        else:
            raise ValueError(f"UNKNOWN_ARM: {arm}")

        self.last_anchor_proposals = None
        self.last_coarse_trajectories = None
        self.last_selected_indices = None
        self._training_target = None

    def forward(self, encoded_tokens: torch.Tensor, past_loc: torch.Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(torch.cat([recent, pooled, self.motion_features(past_loc)], dim=-1))
        scale = F.softplus(self.proposal_scale(context)) + 0.25

        mode_input_base = context[:, :, None] + self.mode_embedding[None, None]
        mode_base = self.mode_mlp(mode_input_base)

        base_anchors_focal = scale[:, 0, None, None] * self.anchor_templates[None]
        geo_feat_0 = (
            compute_anchor_geo_features_12d(base_anchors_focal, past_loc[:, 0])
            if self.coordinator.use_geo else None
        )
        z_focal_updated = self.coordinator(mode_input_base[:, 0], context[:, 0], geo_feat_0)
        mode_delta_focal = self.mode_mlp(z_focal_updated)
        mode_for_delta = torch.cat([mode_delta_focal.unsqueeze(1), mode_base[:, 1:]], dim=1)

        delta = 0.75 * torch.tanh(self.proposal_delta(mode_for_delta)).reshape(
            *mode_for_delta.shape[:-1], self.anchor_count, 2
        )
        relative = scale[:, :, None, None] * (self.anchor_templates[None, None] + delta)
        coarse_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, relative)
        coarse = past_loc[:, :, None, -1:] + coarse_relative

        logits = self.score_head(torch.cat([mode_base, relative.flatten(-2)], dim=-1)).squeeze(-1)
        if self.rel_scorer is not None:
            rel_logit_focal = self.rel_scorer(mode_base[:, 0], relative[:, 0], past_loc[:, 0])
            logits = torch.cat([logits[:, :1] + rel_logit_focal.unsqueeze(1), logits[:, 1:]], dim=1)

        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + relative).detach()
        self.last_coarse_trajectories = coarse.detach()
        self.last_selected_indices = None
        return coarse, logits
