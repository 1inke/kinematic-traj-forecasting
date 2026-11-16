"""Lightweight non-autoregressive trajectory decoders."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def dct_basis(length: int, components: int) -> torch.Tensor:
    """Return the first orthonormal DCT-II basis vectors [T,K]."""
    if components <= 0 or components > length:
        raise ValueError(
            f"components must be in [1, {length}], got {components}"
        )
    time = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.arange(components, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos(
        math.pi * (time + 0.5) * frequency / float(length)
    )
    basis[:, 0] *= math.sqrt(1.0 / float(length))
    if components > 1:
        basis[:, 1:] *= math.sqrt(2.0 / float(length))
    return basis


class CVResidualDCTDecoder(nn.Module):
    """Predict a smooth DCT residual around a constant-velocity trajectory.

    All future coordinates are reconstructed in one parallel operation:

        future = constant_velocity(past) + B_dct @ coefficients(context)

    The final coefficient layer starts near zero, so the initial prediction is
    close to the explicit constant-velocity prior without blocking gradients.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pred_len: int,
        output_dim: int = 2,
        num_basis: int = 6,
        coefficient_init_std: float = 1e-3,
    ):
        super().__init__()
        if output_dim != 2:
            raise ValueError(
                "CVResidualDCTDecoder currently supports 2D trajectories."
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.pred_len = int(pred_len)
        self.output_dim = int(output_dim)
        self.num_basis = int(num_basis)

        self.norm = nn.LayerNorm(self.input_dim, eps=1e-6)
        self.context_projection = nn.Linear(
            self.input_dim, self.hidden_dim
        )
        self.activation = nn.GELU()
        self.coefficient_head = nn.Linear(
            self.hidden_dim, self.num_basis * self.output_dim
        )
        nn.init.xavier_normal_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        nn.init.normal_(
            self.coefficient_head.weight,
            mean=0.0,
            std=float(coefficient_init_std),
        )
        nn.init.zeros_(self.coefficient_head.bias)
        self.register_buffer(
            "basis",
            dct_basis(self.pred_len, self.num_basis),
            persistent=True,
        )

    @staticmethod
    def constant_velocity_prior(
        past_loc: torch.Tensor, pred_len: int
    ) -> torch.Tensor:
        if past_loc.ndim != 4 or past_loc.shape[-1] != 2:
            raise ValueError(
                "past_loc must have shape [batch, agents, history, 2]"
            )
        if past_loc.shape[-2] < 2:
            raise ValueError(
                "At least two observed positions are required for CV."
            )
        last = past_loc[:, :, -1]
        displacement = past_loc[:, :, -1] - past_loc[:, :, -2]
        steps = torch.arange(
            1,
            pred_len + 1,
            dtype=past_loc.dtype,
            device=past_loc.device,
        ).view(1, 1, pred_len, 1)
        return last[:, :, None, :] + steps * displacement[:, :, None, :]

    def forward(
        self, encoded_history: torch.Tensor, past_loc: torch.Tensor
    ) -> torch.Tensor:
        if encoded_history.ndim != 3:
            raise ValueError(
                "encoded_history must have shape [batch, agents, features]"
            )
        if encoded_history.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected decoder input {self.input_dim}, got "
                f"{encoded_history.shape[-1]}"
            )
        hidden = self.activation(
            self.context_projection(self.norm(encoded_history))
        )
        coefficients = self.coefficient_head(hidden).reshape(
            *encoded_history.shape[:-1],
            self.num_basis,
            self.output_dim,
        )
        residual = torch.einsum(
            "tk,...kd->...td",
            self.basis.to(
                device=encoded_history.device,
                dtype=encoded_history.dtype,
            ),
            coefficients,
        )
        return self.constant_velocity_prior(
            past_loc, self.pred_len
        ) + residual


class PCATrajectoryDecoder(nn.Module):
    """One-shot trajectory reconstruction from fixed train-only PCA bases.

    The network predicts k coefficients in a heading-aligned, last-position
    relative coordinate frame. A fixed PCA mean and basis reconstruct all
    future points at once, after which the trajectory is rotated back and
    anchored at the last observed position.
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        num_components: int,
        basis_path: str,
        frequency: float,
        heading_speed_eps: float = 0.05,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.pred_len = int(pred_len)
        self.output_dim = 2
        self.num_components = int(num_components)
        self.frequency = float(frequency)
        self.heading_speed_eps = float(heading_speed_eps)

        basis_file = Path(basis_path)
        if not basis_file.exists():
            raise FileNotFoundError(f"PCA basis not found: {basis_file}")
        with np.load(basis_file, allow_pickle=False) as payload:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            components = np.asarray(
                payload["components"], dtype=np.float32
            )
            stored_pred_len = int(payload["pred_len"])
            coordinate_frame = str(payload["coordinate_frame"])
        expected_dimension = self.pred_len * self.output_dim
        if stored_pred_len != self.pred_len:
            raise ValueError(
                f"PCA basis predicts {stored_pred_len} steps, expected "
                f"{self.pred_len}."
            )
        if mean.shape != (expected_dimension,):
            raise ValueError(f"Unexpected PCA mean shape: {mean.shape}")
        if components.shape[1] != expected_dimension:
            raise ValueError(
                f"Unexpected PCA component shape: {components.shape}"
            )
        if not 1 <= self.num_components <= components.shape[0]:
            raise ValueError(
                f"num_components must be in [1, {components.shape[0]}]"
            )
        if coordinate_frame not in {"relative", "heading_aligned"}:
            raise ValueError(
                f"Unsupported PCA coordinate frame: {coordinate_frame}"
            )
        self.coordinate_frame = coordinate_frame

        self.coefficient_head = nn.Linear(
            self.input_dim, self.num_components
        )
        self.register_buffer(
            "pca_mean", torch.from_numpy(mean), persistent=True
        )
        self.register_buffer(
            "pca_components",
            torch.from_numpy(components[: self.num_components]),
            persistent=True,
        )

    def _heading(self, past_loc: torch.Tensor):
        velocity = (
            past_loc[:, :, -1] - past_loc[:, :, -2]
        ) * self.frequency
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        angle = torch.atan2(velocity[..., 1], velocity[..., 0])
        angle = torch.where(
            speed >= self.heading_speed_eps,
            angle,
            torch.zeros_like(angle),
        )
        return torch.cos(angle), torch.sin(angle)

    def forward(
        self, encoded_history: torch.Tensor, past_loc: torch.Tensor
    ) -> torch.Tensor:
        if encoded_history.ndim != 3:
            raise ValueError(
                "encoded_history must have shape [batch, agents, features]"
            )
        if encoded_history.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected decoder input {self.input_dim}, got "
                f"{encoded_history.shape[-1]}"
            )
        coefficients = self.coefficient_head(encoded_history)
        local_flat = self.pca_mean.to(
            dtype=encoded_history.dtype
        ) + torch.einsum(
            "...k,kd->...d",
            coefficients,
            self.pca_components.to(dtype=encoded_history.dtype),
        )
        local = local_flat.reshape(
            *encoded_history.shape[:-1],
            self.pred_len,
            self.output_dim,
        )
        if self.coordinate_frame == "relative":
            relative = local
        else:
            cosine, sine = self._heading(past_loc)
            local_x = local[..., 0]
            local_y = local[..., 1]
            relative = torch.empty_like(local)
            relative[..., 0] = (
                cosine[..., None] * local_x
                - sine[..., None] * local_y
            )
            relative[..., 1] = (
                sine[..., None] * local_x
                + cosine[..., None] * local_y
            )
        return past_loc[:, :, -1:, :] + relative


class EndpointPCATrajectoryDecoder(nn.Module):
    """Predict an endpoint and a low-rank, endpoint-zero path residual."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        num_components: int,
        basis_path: str,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.pred_len = int(pred_len)
        self.output_dim = 2
        self.num_components = int(num_components)
        basis_file = Path(basis_path)
        if not basis_file.exists():
            raise FileNotFoundError(
                f"Endpoint-PCA basis not found: {basis_file}"
            )
        with np.load(basis_file, allow_pickle=False) as payload:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            components = np.asarray(
                payload["components"], dtype=np.float32
            )
            stored_pred_len = int(payload["pred_len"])
            residual_steps = int(payload["residual_steps"])
            endpoint_constrained = bool(payload["endpoint_constrained"])
        expected_shape_dimension = (self.pred_len - 1) * self.output_dim
        if stored_pred_len != self.pred_len:
            raise ValueError(
                f"Basis predicts {stored_pred_len} steps, expected "
                f"{self.pred_len}."
            )
        if residual_steps != self.pred_len - 1:
            raise ValueError("Unexpected endpoint-PCA residual length.")
        if not endpoint_constrained:
            raise ValueError("PCA basis is not endpoint constrained.")
        if mean.shape != (expected_shape_dimension,):
            raise ValueError(f"Unexpected residual mean: {mean.shape}")
        if components.shape[1] != expected_shape_dimension:
            raise ValueError(
                f"Unexpected residual basis: {components.shape}"
            )
        if not 1 <= self.num_components <= components.shape[0]:
            raise ValueError(
                f"num_components must be in [1, {components.shape[0]}]"
            )

        self.parameter_head = nn.Linear(
            self.input_dim, self.output_dim + self.num_components
        )
        self.register_buffer(
            "residual_mean", torch.from_numpy(mean), persistent=True
        )
        self.register_buffer(
            "residual_components",
            torch.from_numpy(components[: self.num_components]),
            persistent=True,
        )
        self.register_buffer(
            "endpoint_alpha",
            torch.arange(1, self.pred_len + 1, dtype=torch.float32)
            / float(self.pred_len),
            persistent=True,
        )

    def forward(
        self, encoded_history: torch.Tensor, past_loc: torch.Tensor
    ) -> torch.Tensor:
        if encoded_history.ndim != 3:
            raise ValueError(
                "encoded_history must have shape [batch, agents, features]"
            )
        parameters = self.parameter_head(encoded_history)
        endpoint = parameters[..., : self.output_dim]
        coefficients = parameters[..., self.output_dim :]
        residual_flat = self.residual_mean.to(
            dtype=encoded_history.dtype
        ) + torch.einsum(
            "...k,kd->...d",
            coefficients,
            self.residual_components.to(dtype=encoded_history.dtype),
        )
        residual_prefix = residual_flat.reshape(
            *encoded_history.shape[:-1],
            self.pred_len - 1,
            self.output_dim,
        )
        residual = torch.cat(
            [
                residual_prefix,
                torch.zeros_like(residual_prefix[..., :1, :]),
            ],
            dim=-2,
        )
        # endpoint has [...,2]; insert time before coordinates.
        line = self.endpoint_alpha.to(
            dtype=encoded_history.dtype
        ).view(*([1] * (endpoint.ndim - 1)), self.pred_len, 1)
        relative = line * endpoint.unsqueeze(-2) + residual
        return past_loc[:, :, -1:, :] + relative


class ModeQueryDecoderLayer(nn.Module):
    """One parallel self/cross-attention update for trajectory modes."""

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        use_mode_self_attention: bool,
    ):
        super().__init__()
        self.self_norm = (
            nn.LayerNorm(query_dim, eps=1e-6)
            if use_mode_self_attention
            else None
        )
        self.self_attention = (
            nn.MultiheadAttention(
                query_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            if use_mode_self_attention
            else None
        )
        self.cross_norm = nn.LayerNorm(query_dim, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(query_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, query_dim),
        )

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        if self.self_attention is not None:
            normalized = self.self_norm(queries)
            attended, _ = self.self_attention(
                normalized, normalized, normalized, need_weights=False
            )
            queries = queries + attended
        attended, _ = self.cross_attention(
            self.cross_norm(queries),
            memory,
            memory,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class ModeQueryEndpointPCADecoder(nn.Module):
    """Cross-attentive mode queries with endpoint-constrained PCA paths.

    One query represents one complete future trajectory.  Queries read the
    per-agent encoded history in parallel and predict an explicit endpoint
    plus coefficients of a residual basis whose final residual is zero.
    """

    def __init__(
        self,
        token_dim: int,
        query_dim: int,
        pred_len: int,
        num_modes: int,
        num_components: int,
        basis_path: str,
        num_heads: int = 4,
        ffn_dim: int = 64,
        dropout: float = 0.1,
        use_mode_self_attention: bool = False,
        num_layers: int = 1,
        use_residual_refiner: bool = False,
        refiner_hidden_dim: int = 64,
        refiner_max_offset: float = 0.35,
        score_hidden_dim: int = 64,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.query_dim = int(query_dim)
        self.pred_len = int(pred_len)
        self.output_dim = 2
        self.num_modes = int(num_modes)
        self.num_components = int(num_components)
        self.use_mode_self_attention = bool(use_mode_self_attention)
        self.num_layers = int(num_layers)
        self.use_residual_refiner = bool(use_residual_refiner)
        self.refiner_max_offset = float(refiner_max_offset)
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.query_dim % int(num_heads):
            raise ValueError("query_dim must be divisible by num_heads.")

        basis_file = Path(basis_path)
        if not basis_file.exists():
            raise FileNotFoundError(
                f"Endpoint-PCA basis not found: {basis_file}"
            )
        with np.load(basis_file, allow_pickle=False) as payload:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            components = np.asarray(
                payload["components"], dtype=np.float32
            )
            stored_pred_len = int(payload["pred_len"])
            residual_steps = int(payload["residual_steps"])
            endpoint_constrained = bool(payload["endpoint_constrained"])
        expected_dimension = (self.pred_len - 1) * self.output_dim
        if stored_pred_len != self.pred_len:
            raise ValueError(
                f"Basis predicts {stored_pred_len} steps, expected "
                f"{self.pred_len}."
            )
        if residual_steps != self.pred_len - 1:
            raise ValueError("Unexpected endpoint-PCA residual length.")
        if not endpoint_constrained:
            raise ValueError("PCA basis is not endpoint constrained.")
        if mean.shape != (expected_dimension,):
            raise ValueError(f"Unexpected residual mean: {mean.shape}")
        if components.shape[1] != expected_dimension:
            raise ValueError(
                f"Unexpected residual basis: {components.shape}"
            )
        if not 1 <= self.num_components <= components.shape[0]:
            raise ValueError(
                f"num_components must be in [1, {components.shape[0]}]"
            )

        self.memory_projection = nn.Linear(self.token_dim, self.query_dim)
        self.memory_norm = nn.LayerNorm(self.query_dim, eps=1e-6)
        self.context_to_query = nn.Linear(self.query_dim, self.query_dim)
        self.mode_queries = nn.Parameter(
            torch.empty(self.num_modes, self.query_dim)
        )
        nn.init.normal_(self.mode_queries, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [
                ModeQueryDecoderLayer(
                    query_dim=self.query_dim,
                    num_heads=int(num_heads),
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                    use_mode_self_attention=self.use_mode_self_attention,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.parameter_head = nn.Linear(
            self.query_dim, self.output_dim + self.num_components
        )
        parameter_dim = self.output_dim + self.num_components
        self.score_head = nn.Sequential(
            nn.Linear(
                self.query_dim + 2 * parameter_dim,
                int(score_hidden_dim),
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(score_hidden_dim), 1),
        )
        if self.use_residual_refiner:
            self.residual_refiner = nn.Sequential(
                nn.Linear(self.query_dim, int(refiner_hidden_dim)),
                nn.GELU(),
                nn.Linear(
                    int(refiner_hidden_dim),
                    self.pred_len * self.output_dim,
                ),
            )
            nn.init.zeros_(self.residual_refiner[-1].weight)
            nn.init.zeros_(self.residual_refiner[-1].bias)
        else:
            self.residual_refiner = None

        self.register_buffer(
            "residual_mean", torch.from_numpy(mean), persistent=True
        )
        self.register_buffer(
            "residual_components",
            torch.from_numpy(components[: self.num_components]),
            persistent=True,
        )
        self.register_buffer(
            "endpoint_alpha",
            torch.arange(1, self.pred_len + 1, dtype=torch.float32)
            / float(self.pred_len),
            persistent=True,
        )

    def _reconstruct(
        self, parameters: torch.Tensor, past_loc: torch.Tensor
    ) -> torch.Tensor:
        endpoint = parameters[..., : self.output_dim]
        coefficients = parameters[..., self.output_dim :]
        residual_flat = self.residual_mean.to(
            dtype=parameters.dtype
        ) + torch.einsum(
            "...k,kd->...d",
            coefficients,
            self.residual_components.to(dtype=parameters.dtype),
        )
        residual_prefix = residual_flat.reshape(
            *parameters.shape[:-1],
            self.pred_len - 1,
            self.output_dim,
        )
        residual = torch.cat(
            [
                residual_prefix,
                torch.zeros_like(residual_prefix[..., :1, :]),
            ],
            dim=-2,
        )
        alpha = self.endpoint_alpha.to(dtype=parameters.dtype).view(
            *([1] * (endpoint.ndim - 1)), self.pred_len, 1
        )
        relative = alpha * endpoint.unsqueeze(-2) + residual
        return past_loc[:, :, None, -1:, :] + relative

    def forward(
        self, encoded_tokens: torch.Tensor, past_loc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded_tokens.ndim != 4:
            raise ValueError(
                "encoded_tokens must have shape [batch, agents, time, dim]"
            )
        if encoded_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"Expected token dim {self.token_dim}, got "
                f"{encoded_tokens.shape[-1]}"
            )
        batch_size, agents, history, _ = encoded_tokens.shape
        memory = self.memory_projection(encoded_tokens).reshape(
            batch_size * agents, history, self.query_dim
        )
        memory = self.memory_norm(memory)
        pooled = memory.mean(dim=1)
        queries = self.mode_queries.unsqueeze(0).expand(
            batch_size * agents, -1, -1
        )
        queries = queries + self.context_to_query(pooled).unsqueeze(1)

        for layer in self.layers:
            queries = layer(queries, memory)
        queries = queries.reshape(
            batch_size, agents, self.num_modes, self.query_dim
        )
        parameters = self.parameter_head(queries)
        trajectories = self._reconstruct(parameters, past_loc)
        if self.residual_refiner is not None:
            residual = self.residual_refiner(queries).reshape(
                batch_size,
                agents,
                self.num_modes,
                self.pred_len,
                self.output_dim,
            )
            alpha = self.endpoint_alpha.to(
                dtype=residual.dtype
            ).view(1, 1, 1, self.pred_len, 1)
            envelope = torch.sin(torch.pi * alpha)
            trajectories = trajectories + (
                self.refiner_max_offset
                * torch.tanh(residual)
                * envelope
            )
        centered_parameters = (
            parameters - parameters.mean(dim=-2, keepdim=True)
        )
        score_features = torch.cat(
            [queries, parameters, centered_parameters], dim=-1
        )
        logits = self.score_head(score_features).squeeze(-1)
        if self.num_modes == 1:
            trajectories = trajectories.squeeze(-3)
        return trajectories, logits
