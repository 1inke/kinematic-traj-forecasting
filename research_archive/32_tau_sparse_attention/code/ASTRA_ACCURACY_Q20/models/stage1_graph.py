"""Graph construction utilities for the Stage-1 (G0--G4) experiments.

The edge convention follows the design document: adjacency[b, i, j] == True
means that pedestrian i attends to pedestrian j.  Graphs are built once from
the complete observation window and are then shared by all observed tokens.
"""

from dataclasses import dataclass
import math

import torch
import tau_geometry


@dataclass
class GraphStats:
    edge_count: torch.Tensor
    valid_nodes: torch.Tensor
    mean_out_degree: torch.Tensor
    max_out_degree: torch.Tensor
    density: torch.Tensor


@dataclass
class GraphContext:
    """Complete graph description used by DRSE-Lite.

    All pairwise tensors follow the convention ``[batch, source, target, ...]``
    where source ``i`` attends to target ``j``.
    """

    adjacency: torch.Tensor
    valid: torch.Tensor
    candidates: torch.Tensor
    risk_score: torch.Tensor
    risk_history: torch.Tensor
    kinematic_features: torch.Tensor
    node_features: torch.Tensor
    edge_structure_features: torch.Tensor


class Stage1GraphBuilder:
    MODES = {
        "legacy",
        "full",
        "distance_topk",
        "kinematic_topk",
        "adaptive_kinematic",
    }

    def __init__(self, cfg):
        graph_cfg = getattr(cfg.MODEL, "GRAPH", None)
        self.mode = getattr(graph_cfg, "MODE", "legacy")
        if self.mode not in self.MODES:
            raise ValueError(
                f"Unknown graph mode {self.mode!r}; expected one of "
                f"{sorted(self.MODES)}"
            )

        self.top_k = int(getattr(graph_cfg, "TOP_K", 4))
        self.symmetric = bool(getattr(graph_cfg, "SYMMETRIC", False))
        self.radius = float(getattr(graph_cfg, "RADIUS", 6.0))
        self.k_min = int(getattr(graph_cfg, "K_MIN", 2))
        self.k_max = int(getattr(graph_cfg, "K_MAX", 6))
        self.score_threshold = float(
            getattr(graph_cfg, "SCORE_THRESHOLD", 0.35)
        )
        self.gamma = float(getattr(graph_cfg, "HISTORY_GAMMA", 0.25))
        self.horizon = float(getattr(graph_cfg, "DCA_HORIZON", 4.8))
        self.ttc_max = float(getattr(graph_cfg, "TTC_MAX", 8.0))
        data_cfg = getattr(cfg, "DATA", None)
        self.frequency = float(getattr(data_cfg, "FREQUENCY", 1.0))
        self.eps = 1e-6

        weights = getattr(graph_cfg, "SCORE_WEIGHTS", None)
        self.lambda_d = float(getattr(weights, "DISTANCE", 0.25))
        self.lambda_c = float(getattr(weights, "CLOSING_SPEED", 0.20))
        self.lambda_t = float(getattr(weights, "TTC", 0.20))
        self.lambda_a = float(getattr(weights, "ANGLE", 0.15))
        self.lambda_dca = float(getattr(weights, "DCA", 0.20))

        scales = getattr(graph_cfg, "SCORE_SCALES", None)
        self.sigma_d = float(getattr(scales, "DISTANCE", 2.0))
        self.sigma_t = float(getattr(scales, "TTC", 2.0))
        self.sigma_dca = float(getattr(scales, "DCA", 1.0))
        self.closing_speed_scale = float(
            getattr(scales, "CLOSING_SPEED", 1.5)
        )
        structure_cfg = getattr(cfg.MODEL, "STRUCTURE", None)
        rbf_cfg = getattr(structure_cfg, "RBF", None)
        self.rbf_centers = int(getattr(rbf_cfg, "NUM_CENTERS", 8))
        if self.rbf_centers < 2:
            raise ValueError("RBF.NUM_CENTERS must be at least 2")
        self.rbf_bandwidth = float(
            getattr(
                rbf_cfg,
                "BANDWIDTH",
                1.0 / (self.rbf_centers - 1),
            )
        )
        if self.rbf_bandwidth <= 0:
            raise ValueError("RBF.BANDWIDTH must be positive")
        self.use_rbf = bool(getattr(structure_cfg, "USE_RBF", True))
        # 16 raw last/history channels, four explicit bearing channels,
        # and 4 variables x 2 temporal summaries x C RBF channels.
        self.kinematic_feature_dim = 20 + 8 * self.rbf_centers
        # Six binary/count motifs and their six risk-weighted counterparts.
        self.motif_feature_dim = 12

    @staticmethod
    def valid_mask(num_valid, batch_size, num_nodes, device):
        if num_valid is None:
            return torch.ones(
                batch_size, num_nodes, dtype=torch.bool, device=device
            )
        counts = torch.as_tensor(num_valid, device=device).long().view(-1)
        if counts.numel() == 1 and batch_size > 1:
            counts = counts.expand(batch_size)
        if counts.numel() != batch_size:
            raise ValueError(
                f"num_valid has {counts.numel()} values for batch size "
                f"{batch_size}"
            )
        counts = counts.clamp(min=1, max=num_nodes)
        return (
            torch.arange(num_nodes, device=device).unsqueeze(0)
            < counts.unsqueeze(1)
        )

    def _history_components(self, trajectories):
        # trajectories: [B, T, N, 2]
        velocity = torch.zeros_like(trajectories)
        velocity[:, 1:] = (
            trajectories[:, 1:] - trajectories[:, :-1]
        ) * self.frequency
        velocity[:, 0] = velocity[:, 1] if trajectories.shape[1] > 1 else 0

        # Row i, column j contains p_j - p_i and v_j - v_i.
        delta_p = trajectories.unsqueeze(2) - trajectories.unsqueeze(3)
        delta_v = velocity.unsqueeze(2) - velocity.unsqueeze(3)
        distance = torch.linalg.vector_norm(delta_p, dim=-1)
        closing = torch.relu(
            -(delta_p * delta_v).sum(dim=-1) / (distance + self.eps)
        )
        rel_speed_sq = (delta_v * delta_v).sum(dim=-1)
        closest_time = torch.clamp(
            -(delta_p * delta_v).sum(dim=-1)
            / (rel_speed_sq + self.eps),
            min=0.0,
            max=self.horizon,
        )
        dca = torch.linalg.vector_norm(
            delta_p + closest_time.unsqueeze(-1) * delta_v, dim=-1
        )
        ttc = torch.clamp(
            distance / (closing + self.eps), max=self.ttc_max
        )

        direction = velocity / (
            torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
            + self.eps
        )
        bearing = delta_p / (distance.unsqueeze(-1) + self.eps)
        angle = torch.relu(
            (direction.unsqueeze(3) * bearing).sum(dim=-1)
        )
        lateral_direction = torch.stack(
            [-direction[..., 1], direction[..., 0]], dim=-1
        )
        relative_position_parallel = (
            direction.unsqueeze(3) * delta_p
        ).sum(dim=-1)
        relative_position_lateral = (
            lateral_direction.unsqueeze(3) * delta_p
        ).sum(dim=-1)
        relative_velocity_parallel = (
            direction.unsqueeze(3) * delta_v
        ).sum(dim=-1)
        relative_velocity_lateral = (
            lateral_direction.unsqueeze(3) * delta_v
        ).sum(dim=-1)
        return {
            "distance": distance,
            "closing": closing,
            "ttc": ttc,
            "dca": dca,
            "angle": angle,
            "relative_position_parallel": relative_position_parallel,
            "relative_position_lateral": relative_position_lateral,
            "relative_velocity_parallel": relative_velocity_parallel,
            "relative_velocity_lateral": relative_velocity_lateral,
        }

    def _pairwise_history(self, trajectories):
        components = self._history_components(trajectories)
        return (
            components["distance"],
            components["closing"],
            components["ttc"],
            components["dca"],
            components["angle"],
        )

    def _risk_history(self, components):
        return (
            self.lambda_d
            * torch.exp(-components["distance"] / self.sigma_d)
            + self.lambda_c
            * torch.clamp(
                components["closing"] / self.closing_speed_scale, max=1.0
            )
            + self.lambda_t
            * torch.exp(-components["ttc"] / self.sigma_t)
            + self.lambda_a * components["angle"]
            + self.lambda_dca
            * torch.exp(-components["dca"] / self.sigma_dca)
        )

    def _history_weights(self, length, dtype, device):
        time = torch.arange(length, dtype=dtype, device=device)
        return torch.softmax(self.gamma * time, dim=0)

    def _rbf(self, values):
        """Expand normalized scalars into fixed Gaussian basis functions."""
        centers = torch.linspace(
            0.0,
            1.0,
            self.rbf_centers,
            dtype=values.dtype,
            device=values.device,
        )
        delta = values.clamp(0.0, 1.0).unsqueeze(-1) - centers
        return torch.exp(
            -0.5 * (delta / self.rbf_bandwidth) ** 2
        )

    def _kinematic_features(self, components):
        """Return raw plus RBF-expanded last/history edge features."""
        radius_scale = max(self.radius, self.eps)
        speed_scale = max(self.closing_speed_scale, self.eps)
        instantaneous = torch.stack(
            [
                components["distance"] / radius_scale,
                components["relative_position_parallel"] / radius_scale,
                components["relative_position_lateral"] / radius_scale,
                components["relative_velocity_parallel"] / speed_scale,
                components["relative_velocity_lateral"] / speed_scale,
                components["closing"] / speed_scale,
                components["ttc"] / max(self.ttc_max, self.eps),
                components["dca"] / radius_scale,
            ],
            dim=-1,
        )
        weights = self._history_weights(
            instantaneous.shape[1],
            instantaneous.dtype,
            instantaneous.device,
        )
        history = (
            instantaneous
            * weights.view(1, -1, 1, 1, 1)
        ).sum(dim=1)

        distance = components["distance"] / radius_scale
        closing = components["closing"] / speed_scale
        ttc = components["ttc"] / max(self.ttc_max, self.eps)
        dca = components["dca"] / radius_scale
        rbf_instantaneous = torch.cat(
            [
                self._rbf(distance),
                self._rbf(closing),
                self._rbf(ttc),
                self._rbf(dca),
            ],
            dim=-1,
        )
        rbf_history = (
            rbf_instantaneous
            * weights.view(1, -1, 1, 1, 1)
        ).sum(dim=1)
        if not self.use_rbf:
            # Keep the exact S2 tensor/parameter layout for a controlled
            # information ablation. Only the RBF signal is removed.
            rbf_instantaneous = torch.zeros_like(rbf_instantaneous)
            rbf_history = torch.zeros_like(rbf_history)

        bearing_cos = (
            components["relative_position_parallel"]
            / (components["distance"] + self.eps)
        )
        bearing_sin = (
            components["relative_position_lateral"]
            / (components["distance"] + self.eps)
        )
        bearing = torch.stack([bearing_cos, bearing_sin], dim=-1)
        bearing_history = (
            bearing * weights.view(1, -1, 1, 1, 1)
        ).sum(dim=1)

        return torch.cat(
            [
                instantaneous[:, -1],
                history,
                bearing[:, -1],
                bearing_history,
                rbf_instantaneous[:, -1],
                rbf_history,
            ],
            dim=-1,
        )

    def kinematic_score(self, trajectories):
        components = self._history_components(trajectories)
        score_t = self._risk_history(components)
        history_weight = self._history_weights(
            trajectories.shape[1],
            trajectories.dtype,
            trajectories.device,
        )
        return (score_t * history_weight.view(1, -1, 1, 1)).sum(dim=1)

    @staticmethod
    def _symmetrize(adjacency):
        return adjacency | adjacency.transpose(-1, -2)

    def _topk(self, score, candidates, valid, k):
        del valid  # Candidate masking already excludes invalid sources/targets.
        num_nodes = score.shape[-1]
        adjacency = torch.zeros_like(candidates)
        keep = min(max(int(k), 0), num_nodes)
        if keep == 0:
            return adjacency
        masked_score = score.masked_fill(~candidates, float("-inf"))
        values, indices = torch.topk(masked_score, keep, dim=-1)
        selected = torch.isfinite(values)
        adjacency.scatter_(dim=-1, index=indices, src=selected)
        return adjacency

    def _adaptive(self, score, candidates, valid):
        del valid  # Candidate masking already excludes invalid sources/targets.
        num_nodes = score.shape[-1]
        adjacency = torch.zeros_like(candidates)
        keep = min(max(self.k_max, 0), num_nodes)
        if keep == 0:
            return adjacency
        masked_score = score.masked_fill(~candidates, float("-inf"))
        ranked_values, ranked_indices = torch.topk(
            masked_score, keep, dim=-1
        )
        candidate_count = candidates.sum(dim=-1)
        above_threshold = (
            candidates & (score > self.score_threshold)
        ).sum(dim=-1)
        desired = above_threshold.clamp(
            min=self.k_min, max=self.k_max
        )
        desired = torch.minimum(desired, candidate_count)
        rank = torch.arange(
            keep, device=score.device
        ).view(1, 1, -1)
        selected = (
            (rank < desired.unsqueeze(-1))
            & torch.isfinite(ranked_values)
        )
        adjacency.scatter_(
            dim=-1, index=ranked_indices, src=selected
        )
        return adjacency

    def build(self, trajectories, num_valid=None):
        if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
            raise ValueError(
                "trajectories must have shape [batch, time, agents, 2]"
            )
        batch_size, _, num_nodes, _ = trajectories.shape
        valid = self.valid_mask(
            num_valid, batch_size, num_nodes, trajectories.device
        )
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(
            num_nodes, dtype=torch.bool, device=trajectories.device
        ).unsqueeze(0)
        non_self = pair_valid & ~eye

        if self.mode == "legacy":
            adjacency = eye & pair_valid
        elif self.mode == "full":
            adjacency = pair_valid
        else:
            last_position = trajectories[:, -1]
            distance = torch.cdist(last_position, last_position)
            candidates = non_self & (distance < self.radius)
            if self.mode == "distance_topk":
                adjacency = self._topk(
                    -distance, candidates, valid, self.top_k
                )
            else:
                score = self.kinematic_score(trajectories)
                if self.mode == "kinematic_topk":
                    adjacency = self._topk(
                        score, candidates, valid, self.top_k
                    )
                else:
                    adjacency = self._adaptive(score, candidates, valid)
            if self.symmetric:
                adjacency = self._symmetrize(adjacency)
            adjacency = adjacency | (eye & pair_valid)
        return adjacency, valid

    @staticmethod
    def _masked_risk_distribution(risk_score, neighbor_mask):
        """Softmax over selected non-self neighbors without all-masked NaNs."""
        masked_score = risk_score.masked_fill(~neighbor_mask, float("-inf"))
        maximum = masked_score.max(dim=-1, keepdim=True).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        unnormalized = torch.exp(masked_score - maximum)
        unnormalized = torch.where(
            neighbor_mask, unnormalized, torch.zeros_like(unnormalized)
        )
        return unnormalized / unnormalized.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)

    def _temporary_adjacency(
        self, risk_history, distance_history, valid, non_self
    ):
        """Build the document-defined temporary A^t for edge persistence."""
        history = []
        for time_index in range(risk_history.shape[1]):
            candidates_t = non_self & (
                distance_history[:, time_index] < self.radius
            )
            adjacency_t = self._adaptive(
                risk_history[:, time_index],
                candidates_t,
                valid,
            )
            history.append(adjacency_t)
        return torch.stack(history, dim=1)

    def build_context(self, trajectories, num_valid=None):
        """Build G4 and all node/edge descriptors required by DRSE-Lite."""
        if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
            raise ValueError(
                "trajectories must have shape [batch, time, agents, 2]"
            )
        batch_size, obs_len, num_nodes, _ = trajectories.shape
        valid = self.valid_mask(
            num_valid, batch_size, num_nodes, trajectories.device
        )
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(
            num_nodes, dtype=torch.bool, device=trajectories.device
        ).unsqueeze(0)
        non_self = pair_valid & ~eye

        components = self._history_components(trajectories)
        risk_history = self._risk_history(components)
        weights = self._history_weights(
            obs_len, trajectories.dtype, trajectories.device
        )
        risk_score = (
            risk_history * weights.view(1, -1, 1, 1)
        ).sum(dim=1)
        candidates = non_self & (
            components["distance"][:, -1] < self.radius
        )

        if hasattr(self, "tau_radius"):
            distance = components["distance"][:, -1]
            tau = tau_geometry.history(trajectories, self.frequency, self.tau_radius)[:, -1]
            adjacency = tau_geometry.select(tau, distance, candidates, 2.4, self.k_min, self.k_max)
        elif self.mode == "adaptive_kinematic":
            adjacency = self._adaptive(
                risk_score, candidates, valid
            )
        elif self.mode == "kinematic_topk":
            adjacency = self._topk(
                risk_score, candidates, valid, self.top_k
            )
        elif self.mode == "distance_topk":
            adjacency = self._topk(
                -components["distance"][:, -1],
                candidates,
                valid,
                self.top_k,
            )
        elif self.mode == "full":
            adjacency = non_self.clone()
        else:
            adjacency = torch.zeros_like(non_self)

        if self.symmetric:
            adjacency = self._symmetrize(adjacency)
        neighbor_mask = adjacency & non_self
        adjacency = adjacency | (eye & pair_valid)

        normalized_risk = self._masked_risk_distribution(
            risk_score, neighbor_mask
        )
        candidate_count = candidates.sum(dim=-1).to(trajectories.dtype)
        out_degree = neighbor_mask.sum(dim=-1).to(trajectories.dtype)
        in_degree = neighbor_mask.sum(dim=-2).to(trajectories.dtype)
        degree_imbalance = (out_degree - in_degree) / (
            out_degree + in_degree + self.eps
        )
        risk_total = (
            risk_score.masked_fill(~neighbor_mask, 0.0).sum(dim=-1)
        )
        risk_max = risk_score.masked_fill(
            ~neighbor_mask, float("-inf")
        ).max(dim=-1).values
        risk_max = torch.where(
            torch.isfinite(risk_max), risk_max, torch.zeros_like(risk_max)
        )
        risk_entropy = -(
            normalized_risk
            * torch.log(normalized_risk.clamp(min=1e-8))
        ).sum(dim=-1)
        node_features = torch.stack(
            [
                candidate_count / max(num_nodes - 1, 1),
                out_degree / max(num_nodes - 1, 1),
                in_degree / max(num_nodes - 1, 1),
                degree_imbalance,
                risk_total / max(self.k_max, 1),
                risk_max,
                risk_entropy
                / max(math.log(float(max(self.k_max, 2))), self.eps),
            ],
            dim=-1,
        )
        node_features = node_features * valid.unsqueeze(-1)

        del normalized_risk  # Motifs use counts and unnormalized risk strength.
        adjacency_float = neighbor_mask.to(trajectories.dtype)
        edge_strength = risk_score * adjacency_float
        motif_scale = float(max(self.k_max, 1))

        forward_count = torch.matmul(
            adjacency_float, adjacency_float
        ) / motif_scale
        backward_count = forward_count.transpose(-1, -2)
        common_out_count = torch.matmul(
            adjacency_float, adjacency_float.transpose(-1, -2)
        ) / motif_scale
        common_in_count = torch.matmul(
            adjacency_float.transpose(-1, -2), adjacency_float
        ) / motif_scale
        convergence_count = (
            in_degree.unsqueeze(-2).expand(-1, num_nodes, -1)
            / motif_scale
        )
        divergence_count = (
            out_degree.unsqueeze(-1).expand(-1, -1, num_nodes)
            / motif_scale
        )

        forward_strength = torch.matmul(
            edge_strength, edge_strength
        ) / motif_scale
        backward_strength = forward_strength.transpose(-1, -2)
        common_out_strength = torch.matmul(
            edge_strength, edge_strength.transpose(-1, -2)
        ) / motif_scale
        common_in_strength = torch.matmul(
            edge_strength.transpose(-1, -2), edge_strength
        ) / motif_scale
        incoming_strength = edge_strength.sum(dim=-2) / motif_scale
        outgoing_strength = edge_strength.sum(dim=-1) / motif_scale
        convergence_strength = incoming_strength.unsqueeze(-2).expand(
            -1, num_nodes, -1
        )
        divergence_strength = outgoing_strength.unsqueeze(-1).expand(
            -1, -1, num_nodes
        )

        edge_structure = torch.stack(
            [
                forward_count,
                backward_count,
                common_out_count,
                common_in_count,
                convergence_count,
                divergence_count,
                forward_strength,
                backward_strength,
                common_out_strength,
                common_in_strength,
                convergence_strength,
                divergence_strength,
            ],
            dim=-1,
        )
        edge_structure = edge_structure * neighbor_mask.unsqueeze(-1)

        kinematic_features = self._kinematic_features(components)
        kinematic_features = (
            kinematic_features * neighbor_mask.unsqueeze(-1)
        )
        return GraphContext(
            adjacency=adjacency,
            valid=valid,
            candidates=candidates,
            risk_score=risk_score,
            risk_history=risk_history,
            kinematic_features=kinematic_features,
            node_features=node_features,
            edge_structure_features=edge_structure,
        )

    @staticmethod
    def rwpe_skeleton(adjacency):
        return adjacency | adjacency.transpose(-1, -2)

    @staticmethod
    def attention_mask(
        adjacency, obs_len, num_heads, edge_bias=None
    ):
        """Convert adjacency and shared/per-head bias to a Transformer mask.

        Every observed token of i can attend to every observed token of j when
        i -> j exists. Padding nodes receive a computational self-loop so no
        query row is entirely ``-inf`` (which would make softmax return NaN).
        These safety loops are mask-only and do not change graph statistics.
        The returned shape is [B * H, T * N, T * N].
        """
        batch_size, num_nodes, _ = adjacency.shape
        safety_eye = torch.eye(
            num_nodes, dtype=torch.bool, device=adjacency.device
        ).unsqueeze(0)
        safe_adjacency = adjacency | safety_eye
        allowed = (
            safe_adjacency[:, None, :, None, :]
            .expand(-1, obs_len, -1, obs_len, -1)
            .reshape(batch_size, obs_len * num_nodes, obs_len * num_nodes)
        )
        allowed = allowed[:, None].expand(-1, num_heads, -1, -1)
        if edge_bias is None:
            mask = torch.zeros(
                allowed.shape,
                dtype=torch.float32,
                device=adjacency.device,
            )
        else:
            if edge_bias.ndim == 3:
                if edge_bias.shape != adjacency.shape:
                    raise ValueError(
                        "shared edge_bias must have shape [B,N,N]"
                    )
                edge_bias = edge_bias[:, None].expand(
                    -1, num_heads, -1, -1
                )
            elif edge_bias.ndim == 4:
                expected = (
                    batch_size,
                    num_heads,
                    num_nodes,
                    num_nodes,
                )
                if tuple(edge_bias.shape) != expected:
                    raise ValueError(
                        f"per-head edge_bias must have shape {expected}"
                    )
            else:
                raise ValueError(
                    "edge_bias must have shape [B,N,N] or [B,H,N,N]"
                )
            mask = (
                edge_bias[:, :, None, :, None, :]
                .expand(-1, -1, obs_len, -1, obs_len, -1)
                .reshape(
                    batch_size,
                    num_heads,
                    obs_len * num_nodes,
                    obs_len * num_nodes,
                )
                .to(dtype=torch.float32)
            )
        mask.masked_fill_(~allowed, float("-inf"))
        return mask.reshape(
            batch_size * num_heads,
            obs_len * num_nodes,
            obs_len * num_nodes,
        )

    @staticmethod
    def stats(adjacency, valid):
        num_nodes = adjacency.shape[-1]
        eye = torch.eye(
            num_nodes, dtype=torch.bool, device=adjacency.device
        ).unsqueeze(0)
        edges = adjacency & ~eye
        degrees = edges.sum(dim=-1)
        valid_nodes = valid.sum(dim=-1)
        edge_count = edges.sum(dim=(-1, -2))
        denominator = (valid_nodes * (valid_nodes - 1)).clamp(min=1)
        return GraphStats(
            edge_count=edge_count,
            valid_nodes=valid_nodes,
            mean_out_degree=degrees.sum(dim=-1)
            / valid_nodes.clamp(min=1),
            max_out_degree=degrees.max(dim=-1).values,
            density=edge_count / denominator,
        )
