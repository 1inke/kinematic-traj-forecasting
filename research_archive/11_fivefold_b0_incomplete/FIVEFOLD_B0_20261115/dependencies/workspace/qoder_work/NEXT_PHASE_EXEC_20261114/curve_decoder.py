"""E2 curve decoder: same h3 anchors, different joining rule.  (torch; NOT executed in authoring env)

Single factor relative to B0: how the three learned anchors (steps 4/8/12) become a
12-step path.  Context, mode embedding, templates, proposal_scale, proposal_delta
(0.75*tanh), score head and loss are untouched.

Arms
  linear          B0 (identity wrapper, used to check the wrapper is exact)
  natural         cubic spline through (0,a4,a8,a12), S''(0)=S''(12)=0, no velocity input
  clamped         cubic spline, S'(0)=v_obs (last observed per-step displacement), S''(12)=0
  clamped_launch  clamped, with a per-mode bounded launch offset: S'(0)=v_obs+dv_k,
                  dv_k = DV_MAX*tanh(W m_k), W zero-initialised (identical to clamped at init)

Contrasts: natural-linear = smooth joining alone; clamped-natural = observed-velocity
continuity; clamped_launch-clamped = hard vs soft continuity.

Note on reachability: unlike DGeo-V2 KinTaylor (which shrank the anchor residual to
small metric bounds and stayed piecewise linear), every arm here keeps the original
anchor reachable set; only the path between anchors changes.

Usage (inside an E2E runner, after baseline_factory.make_model):
    model, cfg = make_model(device, seed)
    attach_curve_decoder(model, arm)          # before model.to(device) is fine too
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spline_basis import basis  # noqa: E402  (scipy at construction only)

ARMS = ("linear", "natural", "clamped", "clamped_launch")
# FIT-GT junction acceleration p99 = 1.2525 m/s^2 (C0 MOTION_REFERENCE, sealed before B0)
# -> velocity change per 0.4 s step 0.501 m/s -> 0.20 m per step.  Fixed, never tuned on dev.
DV_MAX = 0.20


class CurveAnchorDecoder(nn.Module):
    def __init__(self, base: nn.Module, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        if getattr(base, "use_refiner", False):
            raise ValueError("ONLY_ORIGINAL_H3_NO_REFINER")
        if tuple(base.anchor_steps) != (4, 8, 12) or base.pred_len != 12:
            raise ValueError("EXPECTED_H3_4_8_12")
        # Re-use the ORIGINAL h3 submodules/parameters under their ORIGINAL names
        # (same objects, same init values), so B0 state_dicts load key-for-key.
        _adopt(self, base)
        self.arm = arm
        W, w_v = basis("linear" if arm == "linear" else ("natural" if arm == "natural" else "clamped"))
        self.register_buffer("curve_W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer("curve_wv", torch.tensor(w_v, dtype=torch.float32))
        self.launch = None
        if arm == "clamped_launch":
            # Construct under a forked CPU RNG so shared-parameter init and later dropout
            # streams stay identical across arms of the same seed.
            with torch.random.fork_rng(devices=[]):
                dim = self.mode_embedding.shape[-1]
                self.launch = nn.Linear(dim, 2)
                nn.init.zeros_(self.launch.weight); nn.init.zeros_(self.launch.bias)
        self.diagnostics = {}
        # attributes the surrounding code reads
        for name in ("num_modes", "pred_len", "anchor_steps", "anchor_count", "use_refiner", "top_m"):
            setattr(self, name, getattr(base, name))
        self.last_anchor_proposals = None
        self.last_coarse_trajectories = None
        self.last_selected_indices = None

    def set_training_target(self, target):
        self._training_target = None   # plain h3 without refiner ignores the target

    def forward(self, encoded_tokens: torch.Tensor, past_loc: torch.Tensor):
        b = self
        recent, pooled = encoded_tokens[:, :, -1], encoded_tokens.mean(dim=2)
        context = b.context(torch.cat((recent, pooled, b.motion_features(past_loc)), dim=-1))
        mode = b.mode_mlp(context[:, :, None] + b.mode_embedding[None, None])
        scale = F.softplus(b.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(b.proposal_delta(mode)).reshape(*mode.shape[:-1], b.anchor_count, 2)
        relative = scale[:, :, None, None] * (b.anchor_templates[None, None] + delta)   # [B,A,K,3,2]
        coarse_relative = torch.einsum("tp,bakpc->baktc", self.curve_W, relative)
        if self.arm in ("clamped", "clamped_launch"):
            v_obs = past_loc[:, :, -1] - past_loc[:, :, -2]                         # [B,A,2] m/step
            v0 = v_obs[:, :, None].expand(-1, -1, mode.shape[2], -1)                 # [B,A,K,2]
            if self.launch is not None:
                dv = DV_MAX * torch.tanh(self.launch(mode))
                v0 = v0 + dv
                self.diagnostics = {"launch_norm": torch.linalg.vector_norm(dv, dim=-1).detach()}
            coarse_relative = coarse_relative + self.curve_wv[None, None, None, :, None] * v0[:, :, :, None, :]
        coarse = past_loc[:, :, None, -1:, :] + coarse_relative
        logits = b.score_head(torch.cat((mode, relative.flatten(-2)), dim=-1)).squeeze(-1)
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + relative).detach()
        self.last_coarse_trajectories = coarse.detach()
        return coarse, logits


NEW_KEYS = ("curve_W", "curve_wv", "launch.weight", "launch.bias")


def _adopt(module: nn.Module, base: nn.Module) -> None:
    """Copy h3 children/params/buffers by reference, keeping their names (UDDecoder pattern)."""
    for name, child in base._modules.items():
        module.add_module(name, child)
    for name, param in base._parameters.items():
        module.register_parameter(name, param)
    for name, buf in base._buffers.items():
        module.register_buffer(name, buf)
    module.motion_features = type(base).motion_features   # staticmethod
    module._training_target = None


def load_b0_weights(wrapped_model: nn.Module, b0_state: dict) -> None:
    """Load a B0 state_dict into a wrapped model; only the new E2 keys may be missing."""
    missing, unexpected = wrapped_model.load_state_dict(b0_state, strict=False)
    bad = [k for k in missing if not k.split("decoder.", 1)[-1] in NEW_KEYS] + list(unexpected)
    if bad:
        raise ValueError(f"STATE_KEY_MISMATCH {bad[:5]}")


def attach_curve_decoder(model: nn.Module, arm: str) -> nn.Module:
    model.decoder = CurveAnchorDecoder(model.decoder, arm)
    return model


@torch.no_grad()
def self_check(model_linear_wrapped, model_original, past, num_valid, prediction_fn):
    """Preflight: the 'linear' wrapper must reproduce the original h3 bit-for-bit
    (same weights loaded), and every arm must pass through its anchors at 4/8/12."""
    a, la, *_ = prediction_fn(model_linear_wrapped, past, num_valid)
    b, lb, *_ = prediction_fn(model_original, past, num_valid)
    if not (torch.equal(a, b) and torch.equal(la, lb)):
        raise ValueError("LINEAR_WRAPPER_NOT_EXACT")
