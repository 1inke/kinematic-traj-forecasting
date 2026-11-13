import math
import torch
from torch import nn
from torch.nn import functional as F
from hierarchical_decoder_models import HierarchicalLearnedAnchorDecoder


def compute_anchor_geo_features_12d(relative_anchors, past_loc_focal):
    """Causal Maneuver-12D; displacement/sample, not velocity in m/s."""
    a1,a2,a3=relative_anchors.unbind(dim=2)
    u1=a1/4.0; u2=(a2-a1)/4.0; u3=(a3-a2)/4.0
    short=past_loc_focal[:,-1]-past_loc_focal[:,-2]
    long=(past_loc_focal[:,-1]-past_loc_focal[:,-4])/3.0
    acceleration=short-long; trend=short+.5*acceleration
    deviation=torch.sqrt(((u1-trend[:,None])**2).sum(-1,keepdim=True)+1e-6)
    direction=u3/(torch.sqrt((u3*u3).sum(-1,keepdim=True)+1e-4)+.05)
    alignment=(direction*acceleration[:,None]).sum(-1,keepdim=True)
    return torch.cat((a1,a2,a3,u2-u1,u3-u2,deviation,alignment),dim=-1)


def kinematic_taylor_delta(raw):
    """Full-rank bounded metric residual at three anchors; interpolation is STILL piecewise linear."""
    g=torch.tanh(raw)
    v=.16*g[...,0,:]; a=.14*g[...,1,:]; j=.04*g[...,2,:]
    return torch.stack((v,2*v+a+j,3*v+3*a),dim=-2)


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
            qk_in = self.qk_norm(z - z.mean(dim=1, keepdim=True))
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
        if arm not in ("DK-CenterDiff","DK-ManeuverDiff","DK-KinTaylorDiff","DK-FullDiff"):
            raise ValueError("UNKNOWN_V2_ARM "+arm)
        # Common coordinator tensors are constructed BEFORE adding geo_proj:
        # all common tensors are paired across factor A as well as factor B.
        self.coordinator=GeoModeCoordinator(dim=dim,proj_dim=16,heads=4,use_geo=False,differential=True)
        if arm in ("DK-ManeuverDiff","DK-FullDiff"):
            self.coordinator.geo_proj=nn.Linear(12,dim,bias=True)
            nn.init.xavier_normal_(self.coordinator.geo_proj.weight)
            nn.init.zeros_(self.coordinator.geo_proj.bias)
            self.coordinator.use_geo=True
        self.kinematic = arm in ("DK-KinTaylorDiff","DK-FullDiff")
        self.rel_scorer=None

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

        raw_delta=self.proposal_delta(mode_for_delta).reshape(*mode_for_delta.shape[:-1],self.anchor_count,2)
        if self.kinematic:
            relative=scale[:, :, None, None]*self.anchor_templates[None,None]+kinematic_taylor_delta(raw_delta)
        else:
            delta=.75*torch.tanh(raw_delta)
            relative=scale[:, :, None,None]*(self.anchor_templates[None,None]+delta)
        coarse_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, relative)
        coarse = past_loc[:, :, None, -1:] + coarse_relative

        logits = self.score_head(torch.cat([mode_base, relative.flatten(-2)], dim=-1)).squeeze(-1)

        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + relative).detach()
        self.last_coarse_trajectories = coarse.detach()
        self.last_selected_indices = None
        return coarse, logits

