"""E3 geometry-level set refinement (torch; NOT executed in authoring env).

Pass 1 = unchanged h3 proposal (anchors at 4/8/12).  Pass 2 = one small attention
layer over the K=20 candidates of the same agent whose tokens are built from the
pass-1 GEOMETRY and whose attention logits get a pairwise geometric bias (the same
idea as the encoder's RBF kinematic edge bias, applied to the candidate set).
It outputs a bounded anchor residual and a score residual, both zero-initialised,
so every arm equals B0 at initialisation.  Trained end-to-end with the unchanged B0
loss on the final paths/logits (no extra loss term).

Arms (identical parameter tensors; only the attention mask / bias switch differs)
  self        attention restricted to the diagonal: each candidate sees only itself
              -> controls for "second pass + extra capacity"
  set         full K x K attention + geometric bias            (primary)
  set_nobias  full attention, geometric bias disabled (module kept, output unused)

Difference from earlier work
  D-Gate/D-DiffGate: interaction on mode embeddings BEFORE any geometry exists.
  10-27 B1/B2: post-hoc refiner on a frozen native cache, special keep/set losses;
               the run crashed at train_B1_42, so no result exists.
  E3: interaction on candidate geometry, end-to-end, B0 loss only.

Interpolation of the refined anchors uses `basis_variant` ('linear' = B0;
set to the E2 winner for the later 2x2).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spline_basis import basis  # noqa: E402

ARMS = ("self", "set", "set_nobias")
MAX_SHIFT = 0.30   # residual bound, in units of the agent's proposal scale (per axis)


class GeoSetRefiner(nn.Module):
    def __init__(self, mode_dim=64, width=24, heads=4, arm="set"):
        super().__init__()
        if arm not in ARMS or width % heads:
            raise ValueError(arm)
        self.arm, self.heads, self.hd = arm, heads, width // heads
        in_dim = mode_dim + 6 + 6 + 1
        self.inp = nn.Sequential(nn.Linear(in_dim, width), nn.LayerNorm(width))
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.n1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.n2 = nn.LayerNorm(width)
        # pairwise geometric bias from [d_a4, d_a8, d_a12] (metres)
        self.bias = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads))
        nn.init.zeros_(self.bias[-1].weight); nn.init.zeros_(self.bias[-1].bias)
        self.anchor_out = nn.Linear(width, 6)
        self.score_out = nn.Linear(width, 1)
        for layer in (self.anchor_out, self.score_out):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.diagnostics = {}

    def forward(self, mode, relative, scale):
        """mode [N,K,D]; relative [N,K,3,2] metres; scale [N,2] -> (relative', dlogit [N,K])"""
        n, k = relative.shape[:2]
        rel_n = relative / scale[:, None, None, :]
        centred = rel_n - rel_n.mean(1, keepdim=True)
        end_dev = torch.linalg.vector_norm(centred[:, :, -1], dim=-1, keepdim=True)
        x = self.inp(torch.cat((mode, rel_n.flatten(-2), centred.flatten(-2), end_dev), -1))
        h = lambda t: t.reshape(n, k, self.heads, self.hd).transpose(1, 2)
        logits = h(self.q(x)) @ h(self.k(x)).transpose(-1, -2) / math.sqrt(self.hd)   # [N,H,K,K]
        if self.arm == "set":
            d = torch.linalg.vector_norm(relative[:, :, None] - relative[:, None], dim=-1)  # [N,K,K,3]
            logits = logits + self.bias(d).permute(0, 3, 1, 2)
        if self.arm == "self":
            eye = torch.eye(k, dtype=torch.bool, device=x.device)
            logits = logits.masked_fill(~eye, float("-inf"))
        att = torch.softmax(logits, -1)
        y = (att @ h(self.v(x))).transpose(1, 2).reshape(n, k, -1)
        x = self.n1(x + self.o(y))
        x = self.n2(x + self.ffn(x))
        shift = MAX_SHIFT * scale[:, None, None, :] * torch.tanh(self.anchor_out(x)).reshape(n, k, 3, 2)
        dlogit = self.score_out(x).squeeze(-1)
        with torch.no_grad():
            off = ~torch.eye(k, dtype=torch.bool, device=x.device)
            self.diagnostics = {
                "shift_norm": torch.linalg.vector_norm(shift, dim=-1).mean(-1),
                "offdiag_attention_mass": (att * off).sum(-1).mean(1) if self.arm != "self" else None,
                "attention_entropy": -(att.clamp_min(1e-12).log() * att).sum(-1).mean(1),
            }
        return relative + shift, dlogit


class SetRefineDecoder(nn.Module):
    def __init__(self, base: nn.Module, arm: str, basis_variant: str = "linear"):
        super().__init__()
        if getattr(base, "use_refiner", False) or tuple(base.anchor_steps) != (4, 8, 12):
            raise ValueError("EXPECTED_PLAIN_H3")
        for name, child in base._modules.items():
            self.add_module(name, child)
        for name, param in base._parameters.items():
            self.register_parameter(name, param)
        for name, buf in base._buffers.items():
            self.register_buffer(name, buf)
        self.motion_features = type(base).motion_features
        self._training_target = None
        self.arm, self.basis_variant = arm, basis_variant
        W, w_v = basis(basis_variant)
        self.register_buffer("curve_W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer("curve_wv", torch.tensor(w_v, dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261114)   # same refiner init in every arm and seed-paired run
            self.refiner = GeoSetRefiner(mode_dim=self.mode_embedding.shape[-1], arm=arm)
        for name in ("num_modes", "pred_len", "anchor_steps", "anchor_count", "use_refiner", "top_m"):
            setattr(self, name, getattr(base, name))
        self.last_anchor_proposals = self.last_coarse_trajectories = self.last_selected_indices = None
        self.last_pass1_relative = None

    def set_training_target(self, target):
        self._training_target = None

    def forward(self, encoded_tokens, past_loc):
        b = self
        recent, pooled = encoded_tokens[:, :, -1], encoded_tokens.mean(dim=2)
        context = b.context(torch.cat((recent, pooled, b.motion_features(past_loc)), dim=-1))
        mode = b.mode_mlp(context[:, :, None] + b.mode_embedding[None, None])
        scale = F.softplus(b.proposal_scale(context)) + 0.25                          # [B,A,2]
        delta = 0.75 * torch.tanh(b.proposal_delta(mode)).reshape(*mode.shape[:-1], b.anchor_count, 2)
        relative = scale[:, :, None, None] * (b.anchor_templates[None, None] + delta)  # pass 1
        B, A, K = mode.shape[:3]
        refined, dlogit = self.refiner(mode.reshape(B * A, K, -1), relative.reshape(B * A, K, 3, 2),
                                       scale.reshape(B * A, 2))
        refined = refined.reshape(B, A, K, 3, 2); dlogit = dlogit.reshape(B, A, K)
        coarse_relative = torch.einsum("tp,bakpc->baktc", self.curve_W, refined)
        if self.basis_variant == "clamped":
            v_obs = past_loc[:, :, -1] - past_loc[:, :, -2]
            coarse_relative = coarse_relative + self.curve_wv[None, None, None, :, None] * v_obs[:, :, None, None, :]
        coarse = past_loc[:, :, None, -1:, :] + coarse_relative
        # score head sees the FINAL geometry; plus set-aware residual
        logits = b.score_head(torch.cat((mode, refined.flatten(-2)), dim=-1)).squeeze(-1) + dlogit
        self.last_pass1_relative = relative.detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + refined).detach()
        self.last_coarse_trajectories = coarse.detach()
        return coarse, logits


def load_b0_weights(wrapped_model, b0_state):
    """Only curve_* buffers and refiner.* parameters may be missing from a B0 state_dict."""
    missing, unexpected = wrapped_model.load_state_dict(b0_state, strict=False)
    bad = [k for k in missing if not (".refiner." in k or k.endswith("curve_W") or k.endswith("curve_wv"))]
    if bad or unexpected:
        raise ValueError(f"STATE_KEY_MISMATCH {(bad + list(unexpected))[:5]}")


def attach_set_refiner(model, arm, basis_variant="linear"):
    model.decoder = SetRefineDecoder(model.decoder, arm, basis_variant)
    return model


def parameter_report(model):
    ref = model.decoder.refiner
    return {"refiner_params": sum(p.numel() for p in ref.parameters()),
            "whole_model_params": sum(p.numel() for p in model.parameters())}
