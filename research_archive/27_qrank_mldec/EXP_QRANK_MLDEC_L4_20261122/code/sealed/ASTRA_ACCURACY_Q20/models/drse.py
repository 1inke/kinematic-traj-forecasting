"""RBF kinematic and directed-motif attention encoders.

This module intentionally contains no RWPE and no static node-token addition.
Both physical and structural relations enter ASTRA through bounded,
head-aware attention-logit biases.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.stage1_graph import GraphContext


@dataclass
class DRSEOutput:
    node_embedding: torch.Tensor
    edge_bias: torch.Tensor
    kinematic_bias: torch.Tensor
    structure_bias: torch.Tensor


class RelationMLP(nn.Module):
    """Small relation encoder with a non-zero feature path."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.xavier_uniform_(self.network[1].weight)
        nn.init.zeros_(self.network[1].bias)
        nn.init.xavier_uniform_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features)


class DRSELite(nn.Module):
    """Encode RBF kinematics and directed motifs as attention biases."""

    KINEMATIC_RAW_DIM = 20
    MOTIF_DIM = 12
    NODE_RAW_DIM = 7
    DIRECTED_NODE_DIM = NODE_RAW_DIM + 2 * MOTIF_DIM

    def __init__(self, cfg, token_dim):
        super().__init__()
        structure_cfg = cfg.MODEL.STRUCTURE
        self.num_heads = int(cfg.MODEL.NHEAD)
        self.variant = str(structure_cfg.VARIANT)
        self.use_kinematic = bool(
            getattr(structure_cfg, "USE_KINEMATIC_BIAS", True)
        )
        self.use_motif = bool(
            getattr(structure_cfg, "USE_STRUCTURE_BIAS", False)
        )
        self.per_head_motif = bool(
            getattr(structure_cfg, "PER_HEAD_MOTIF", False)
        )
        self.use_node = bool(
            getattr(structure_cfg, "USE_NODE_ENCODING", False)
        )

        hidden_dim = int(getattr(structure_cfg, "HIDDEN_DIM", 32))
        dropout = float(getattr(structure_cfg, "DROPOUT", 0.1))
        rbf_cfg = getattr(structure_cfg, "RBF", None)
        centers = int(getattr(rbf_cfg, "NUM_CENTERS", 8))
        kinematic_dim = self.KINEMATIC_RAW_DIM + 8 * centers

        self.kinematic_encoder = RelationMLP(
            kinematic_dim,
            hidden_dim,
            self.num_heads,
            dropout,
        )
        motif_output_dim = self.num_heads if self.per_head_motif else 1
        self.motif_encoder = RelationMLP(
            self.MOTIF_DIM,
            hidden_dim,
            motif_output_dim,
            dropout,
        )
        if self.use_node:
            # This optional module must not perturb initialization of the
            # shared ASTRA backbone in a controlled ablation.
            cpu_rng_state = torch.get_rng_state()
            self.node_encoder = RelationMLP(
                self.DIRECTED_NODE_DIM,
                hidden_dim,
                token_dim,
                float(
                    getattr(structure_cfg, "NODE_DROPOUT", 0.0)
                ),
            )
            self.node_gate = nn.Parameter(torch.zeros(()))
            torch.set_rng_state(cpu_rng_state)
        else:
            self.node_encoder = None

        # Zero gates make the initial model graph-only. The encoders are not
        # zero-initialized, so gate gradients are non-zero on the first step.
        self.kinematic_gate = nn.Parameter(
            torch.zeros(self.num_heads)
        )
        self.motif_gate = nn.Parameter(torch.zeros(self.num_heads))

        if not self.use_kinematic:
            self.kinematic_gate.requires_grad = False
            for parameter in self.kinematic_encoder.parameters():
                parameter.requires_grad = False
        if not self.use_motif:
            self.motif_gate.requires_grad = False
            for parameter in self.motif_encoder.parameters():
                parameter.requires_grad = False

        self.token_dim = int(token_dim)

    @staticmethod
    def _to_head_first(values):
        return values.permute(0, 3, 1, 2).contiguous()

    def directed_node_features(self, context: GraphContext):
        """Aggregate outgoing and incoming motif/risk structure per node."""
        num_nodes = context.valid.shape[-1]
        eye = torch.eye(
            num_nodes,
            dtype=torch.bool,
            device=context.adjacency.device,
        ).unsqueeze(0)
        edges = context.adjacency & ~eye
        edge_values = context.edge_structure_features
        out_degree = edges.sum(dim=-1).clamp(min=1).unsqueeze(-1)
        in_degree = edges.sum(dim=-2).clamp(min=1).unsqueeze(-1)
        outgoing = edge_values.sum(dim=-2) / out_degree
        incoming = edge_values.sum(dim=-3) / in_degree
        features = torch.cat(
            [context.node_features, outgoing, incoming], dim=-1
        )
        return features * context.valid.unsqueeze(-1)

    def forward(self, context: GraphContext):
        batch_size, num_nodes = context.valid.shape
        dtype = context.kinematic_features.dtype
        device = context.kinematic_features.device
        eye = torch.eye(
            num_nodes, dtype=torch.bool, device=device
        ).unsqueeze(0)
        selected_edges = context.adjacency & ~eye
        selected_edges = selected_edges[:, None].to(dtype)

        if self.use_kinematic:
            encoded = torch.tanh(
                self._to_head_first(
                    self.kinematic_encoder(context.kinematic_features)
                )
            )
            gate = torch.tanh(self.kinematic_gate)[None, :, None, None]
            kinematic_bias = encoded * gate * selected_edges
        else:
            kinematic_bias = torch.zeros(
                batch_size,
                self.num_heads,
                num_nodes,
                num_nodes,
                dtype=dtype,
                device=device,
            )

        if self.use_motif:
            encoded = self.motif_encoder(
                context.edge_structure_features
            )
            if not self.per_head_motif:
                encoded = encoded.expand(
                    -1, -1, -1, self.num_heads
                )
            encoded = torch.tanh(self._to_head_first(encoded))
            gate = torch.tanh(self.motif_gate)[None, :, None, None]
            structure_bias = encoded * gate * selected_edges
        else:
            structure_bias = torch.zeros_like(kinematic_bias)

        if self.use_node:
            node_features = self.directed_node_features(context)
            node_embedding = (
                torch.tanh(self.node_gate)
                * torch.tanh(self.node_encoder(node_features))
                * context.valid.unsqueeze(-1)
            )
        else:
            node_embedding = torch.zeros(
                batch_size,
                num_nodes,
                self.token_dim,
                dtype=dtype,
                device=device,
            )
        return DRSEOutput(
            node_embedding=node_embedding,
            edge_bias=kinematic_bias + structure_bias,
            kinematic_bias=kinematic_bias,
            structure_bias=structure_bias,
        )
