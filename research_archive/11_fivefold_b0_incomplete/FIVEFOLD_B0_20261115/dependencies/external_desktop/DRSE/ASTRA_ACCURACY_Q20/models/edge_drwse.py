"""Edge-relative directed alternative-path random-walk encoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EdgeDRWSEOutput:
    residual: torch.Tensor
    features: torch.Tensor
    stats: dict[str, torch.Tensor]


class EdgeAlternativePathDRWSE(nn.Module):
    """Inject directed P2/P3 alternative-path evidence as an edge message.

    The input graph is never changed. For each already-selected directed edge
    i->j, the feature is

        [(P^2)_ij, (P^3_alt)_ij, (P^2)_ji, (P^3_alt)_ji].

    P is row-normalized from the unweighted directed adjacency after removing
    self-loops. P^2 cannot reuse the direct i->j edge because P has no
    self-loops. P^3_alt additionally removes length-three walks that start or
    end with the direct edge, retaining only alternative structural routes.

    A bias-free edge encoder produces a feature-wise gate for the projected
    neighbour token. A zero-initialized scalar residual gate makes the initial
    model exactly equal to the state+RBF baseline. The zero-control variant
    keeps the identical trainable architecture but replaces all structural
    features with zeros.
    """

    feature_dim = 4

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 8,
        zero_control: bool = False,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.zero_control = bool(zero_control)

        self.edge_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.token_dim, bias=False),
        )
        self.neighbour_projection = nn.Linear(
            self.token_dim, self.token_dim, bias=False
        )
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.apply(self._init_weights)
        self.last_stats = None

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    @staticmethod
    def _non_self_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        num_nodes = adjacency.shape[-1]
        eye = torch.eye(
            num_nodes, dtype=torch.bool, device=adjacency.device
        ).unsqueeze(0)
        return adjacency.bool() & ~eye

    def structural_features(
        self, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B,N,N,4] features and the selected non-self edge mask."""
        selected = self._non_self_adjacency(adjacency)
        transition = selected.to(dtype=torch.float32)
        transition = transition / transition.sum(
            dim=-1, keepdim=True
        ).clamp(min=1.0)

        p2 = torch.bmm(transition, transition)
        p3 = torch.bmm(p2, transition)

        # Remove P_ij * (P^2)_jj and (P^2)_ii * P_ij. The walk
        # i->j->i->j is subtracted twice, so add it back once.
        p2_diag = torch.diagonal(p2, dim1=-2, dim2=-1)
        starts_with_direct = transition * p2_diag.unsqueeze(-2)
        ends_with_direct = p2_diag.unsqueeze(-1) * transition
        repeated_direct = (
            transition
            * transition.transpose(-1, -2)
            * transition
        )
        p3_alternative = (
            p3
            - starts_with_direct
            - ends_with_direct
            + repeated_direct
        ).clamp(min=0.0)

        features = torch.stack(
            [
                p2,
                p3_alternative,
                p2.transpose(-1, -2),
                p3_alternative.transpose(-1, -2),
            ],
            dim=-1,
        )
        features = features * selected.unsqueeze(-1)
        if self.zero_control:
            features = torch.zeros_like(features)
        return features, selected

    def forward(
        self, tokens: torch.Tensor, adjacency: torch.Tensor
    ) -> EdgeDRWSEOutput:
        batch_size, token_count, token_dim = tokens.shape
        num_nodes = adjacency.shape[-1]
        if token_dim != self.token_dim:
            raise ValueError(
                f"Expected token dim {self.token_dim}, got {token_dim}"
            )
        if token_count % num_nodes != 0:
            raise ValueError(
                f"Token count {token_count} is not divisible by {num_nodes}"
            )
        obs_len = token_count // num_nodes
        node_tokens = tokens.reshape(
            batch_size, obs_len, num_nodes, token_dim
        )

        features, selected = self.structural_features(adjacency)
        edge_gate = torch.tanh(self.edge_encoder(features))
        neighbour_values = self.neighbour_projection(node_tokens)
        message = torch.einsum(
            "bijd,btjd->btid", edge_gate, neighbour_values
        )
        degree = selected.sum(dim=-1).clamp(min=1).to(tokens.dtype)
        message = message / degree[:, None, :, None]

        gate = torch.tanh(self.residual_gate)
        residual = gate * message
        residual = residual.reshape(batch_size, token_count, token_dim)

        selected_features = features[selected]
        if selected_features.numel():
            feature_abs_mean = selected_features.abs().mean()
            alternative_edge_rate = (
                selected_features.abs().sum(dim=-1) > 0
            ).to(tokens.dtype).mean()
        else:
            feature_abs_mean = tokens.new_zeros(())
            alternative_edge_rate = tokens.new_zeros(())
        stats = {
            "gate": gate.detach(),
            "feature_abs_mean": feature_abs_mean.detach(),
            "alternative_edge_rate": alternative_edge_rate.detach(),
            "residual_abs_mean": residual.detach().abs().mean(),
        }
        self.last_stats = stats
        return EdgeDRWSEOutput(
            residual=residual,
            features=features,
            stats=stats,
        )
