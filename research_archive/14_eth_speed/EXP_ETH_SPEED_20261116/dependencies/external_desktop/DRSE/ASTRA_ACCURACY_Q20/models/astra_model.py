import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from models.vae import VariationalEncoder
from models.vae import ConditionalVariationalEncoder
from models.stage1_graph import Stage1GraphBuilder
from models.drse import DRSELite
from models.edge_drwse import EdgeAlternativePathDRWSE
from models.trajectory_decoder import (
    EndpointPCATrajectoryDecoder,
    ModeQueryEndpointPCADecoder,
    PCATrajectoryDecoder,
)
warnings.filterwarnings('ignore')
try:
    from icecream import ic
    ic.disable()
except ImportError:
    def ic(*args, **kwargs):
        return args[0] if len(args) == 1 else args

class ASTRA_model(nn.Module):
    def __init__(self, cfg):
        super(ASTRA_model, self).__init__()

        self.cfg = cfg
        self.device = self.cfg.device
        self.num_device = len(self.cfg.device_list)
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.batch_size_device = int(self.batch_size // self.num_device) if self.num_device > 0 else self.batch_size

        self.obs_len = int(self.cfg.PREDICTION.OBS_TIME * self.cfg.DATA.FREQUENCY)
        self.pred_len = math.ceil(self.cfg.PREDICTION.PRED_TIME * self.cfg.DATA.FREQUENCY)

        # Set the number of pedestrians and the legacy ASTRA edge index.
        configured_agents = getattr(self.cfg.DATA, "MAX_AGENTS", None)
        if configured_agents is not None:
            self.num_pedestrians = int(configured_agents)
            source, target = torch.where(
                ~torch.eye(self.num_pedestrians, dtype=torch.bool)
            )
            self.edge_index = torch.stack([source, target])
        if self.cfg.DATASET == 'ETH_UCY':
            if configured_agents is not None:
                pass
            elif self.cfg.SUBSET == 'univ':
                self.num_pedestrians = 2
                self.edge_index = torch.tensor([[0, 1], 
                                                [1, 0]], dtype=torch.long)
            elif self.cfg.SUBSET == 'zara01':
                self.num_pedestrians = 3
                self.edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], 
                                                [1, 2, 0, 2, 0, 1]], dtype=torch.long)
            else:
                self.num_pedestrians = 1
                self.edge_index = torch.empty((2, 0), dtype=torch.long)
        elif configured_agents is None:
            self.num_pedestrians = 1
            self.edge_index = torch.empty((2, 0), dtype=torch.long)

        self.graph_builder = Stage1GraphBuilder(cfg)
        self.last_graph_stats = None
        self.last_drse_stats = None
        structure_cfg = getattr(self.cfg.MODEL, "STRUCTURE", None)
        self.drse_enabled = bool(
            getattr(structure_cfg, "ENABLED", False)
        )
        self.use_self_state = bool(
            getattr(structure_cfg, "USE_SELF_STATE", False)
        )
        self.build_self_state = self.use_self_state or bool(
            getattr(structure_cfg, "BUILD_SELF_STATE", False)
        )
        self.use_edge_drwse = bool(
            getattr(structure_cfg, "USE_EDGE_DRWSE", False)
        )
        self.build_edge_drwse = self.use_edge_drwse or bool(
            getattr(structure_cfg, "BUILD_EDGE_DRWSE", False)
        )
        self.edge_drwse_zero_control = bool(
            getattr(structure_cfg, "EDGE_DRWSE_ZERO_CONTROL", False)
        )
        self.use_rwpe = bool(self.cfg.MODEL.USE_SOCIAL) and bool(
            getattr(structure_cfg, "USE_RWPE", True)
        )
        
        # Dimensions of different encodings
        self.input_dim = 2 if self.cfg.DATASET == 'ETH_UCY' else 4
        self.output_dim = 2 if self.cfg.DATASET == 'ETH_UCY' else 4
        self.unet_dim = self.cfg.MODEL.UNET_DIM if self.cfg.MODEL.USE_PRETRAINED_UNET else 0
        self.spatial_dim = self.cfg.MODEL.SPATIAL_DIM
        self.velo_dim = self.cfg.MODEL.VELO_DIM if self.cfg.MODEL.INC_VELO else 0
        self.walk_length = (
            self.cfg.MODEL.RAND_WALK_LEN if self.use_rwpe else 0
        )
        self.temp_dim = self.cfg.MODEL.TEMP_DIM
        self.scene_dim = self.unet_dim + self.temp_dim if self.cfg.MODEL.USE_PRETRAINED_UNET else 0
        self.output_dim = 2 if self.cfg.DATASET == 'ETH_UCY' else 4
        self.edge_criteria = self.cfg.MODEL.EDGE_CRITERIA
        if self.cfg.MODEL.USE_VAE:
            self.latent_dim = self.cfg.MODEL.LATENT_DIM
            self.num_samples = self.cfg.MODEL.K       
            self.vae_model = VariationalEncoder(self.cfg) 
            self.cvae_model = ConditionalVariationalEncoder(self.cfg)
                
        # Linear Projection: XY coordinates
        self.lin_proj_xy = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.spatial_dim),
        )
        self.lin_proj_xy.apply(self.init_weights)
        
        # Linear Projection: UNET features
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            self.lin_proj_unet = nn.Sequential(
                nn.Linear(in_features=self.cfg.MODEL.FEATURE_DIM, out_features=self.unet_dim),
            )
            self.lin_proj_unet.apply(self.init_weights)   
        
        # Linear Projection: Velocity        
        if self.cfg.MODEL.INC_VELO:
            self.lin_proj_velo = nn.Sequential(
                nn.Linear(in_features=1, out_features=self.velo_dim),
            )
            self.lin_proj_velo.apply(self.init_weights)
        
        # Linear Projection: Social Encodings
        if self.use_rwpe:
            self.lin_proj_social = nn.Sequential(
                nn.Linear(in_features=self.walk_length, out_features=self.walk_length),
            )
            self.lin_proj_social.apply(self.init_weights)        
        
        # Agent Aware Transformer Encoder
        self.token_dim = self.spatial_dim + self.temp_dim + self.velo_dim + self.walk_length
        if self.drse_enabled:
            self.drse_encoder = DRSELite(cfg, self.token_dim)
        else:
            self.drse_encoder = None
        self.norm = nn.LayerNorm(self.token_dim, eps = 1e-6)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=self.cfg.MODEL.NHEAD, dim_feedforward=self.cfg.MODEL.DIM_FEEDFORWARD, dropout=self.cfg.MODEL.DROPOUT, activation='gelu', batch_first=True)
        self.agent_aware_trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.MODEL.ENC_LAYERS, norm = self.norm)

        # Scene Aware Transformer Encoder
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            self.unet_trans_norm = nn.LayerNorm(self.unet_dim+self.temp_dim, eps = 1e-6)
            scene_encoder_layer = nn.TransformerEncoderLayer(d_model=self.unet_dim+self.temp_dim, nhead=self.cfg.MODEL.NHEAD, dim_feedforward=self.cfg.MODEL.DIM_FEEDFORWARD, dropout=self.cfg.MODEL.DROPOUT, activation='gelu', batch_first=True)
            self.scene_aware_trans_encoder = nn.TransformerEncoder(scene_encoder_layer, num_layers=self.cfg.MODEL.ENC_LAYERS, norm = self.unet_trans_norm)
         
        # Transformer Masking
        if self.cfg.MODEL.TRANS_MASK:
            self.mask = self.create_custom_mask()
        else:
            self.mask = None
        
        # Lightweight non-autoregressive decoder. D0 reproduces ASTRA's
        # original one-layer absolute-coordinate head. D1 predicts only a
        # smooth low-rank residual around a constant-velocity trajectory.
        self.decoder_input_dim = self.obs_len * (
            self.token_dim + self.scene_dim
        )
        decoder_cfg = getattr(self.cfg.MODEL, "DECODER", None)
        self.decoder_variant = str(
            getattr(decoder_cfg, "VARIANT", "linear")
        )
        if self.decoder_variant == "linear":
            self.decoder = nn.Sequential(
                nn.Linear(
                    in_features=self.decoder_input_dim,
                    out_features=self.pred_len * self.output_dim,
                ),
            )
            self.decoder.apply(self.init_weights)
        elif self.decoder_variant in {
            "pca",
            "endpoint_pca",
            "query_endpoint_pca",
        }:
            if self.cfg.MODEL.USE_VAE:
                raise ValueError(
                    "PCA/query decoder is not enabled for VAE mode."
                )
            # Advance the global RNG exactly as the legacy linear decoder
            # would, while initializing the alternative decoder from an
            # isolated copy. This keeps every later shared module (notably
            # the self-state encoder) identical in the D0/D1 ablation.
            cpu_rng_state = torch.get_rng_state()
            compatibility_decoder = nn.Sequential(
                nn.Linear(
                    in_features=self.decoder_input_dim,
                    out_features=self.pred_len * self.output_dim,
                ),
            )
            compatibility_decoder.apply(self.init_weights)
            post_linear_decoder_rng_state = torch.get_rng_state()
            torch.set_rng_state(cpu_rng_state)
            if self.decoder_variant == "pca":
                self.decoder = PCATrajectoryDecoder(
                    input_dim=self.decoder_input_dim,
                    pred_len=self.pred_len,
                    num_components=int(
                        getattr(decoder_cfg, "PCA_COMPONENTS", 6)
                    ),
                    basis_path=str(
                        getattr(
                            decoder_cfg,
                            "PCA_BASIS_PATH",
                            "assets/pca/basis_heading_aligned.npz",
                        )
                    ),
                    frequency=float(self.cfg.DATA.FREQUENCY),
                    heading_speed_eps=float(
                        getattr(
                            decoder_cfg, "HEADING_SPEED_EPS", 0.05
                        )
                    ),
                )
                self.decoder.coefficient_head.apply(self.init_weights)
            elif self.decoder_variant == "endpoint_pca":
                self.decoder = EndpointPCATrajectoryDecoder(
                    input_dim=self.decoder_input_dim,
                    pred_len=self.pred_len,
                    num_components=int(
                        getattr(decoder_cfg, "PCA_COMPONENTS", 6)
                    ),
                    basis_path=str(
                        getattr(
                            decoder_cfg,
                            "ENDPOINT_PCA_BASIS_PATH",
                            "assets/pca/"
                            "endpoint_residual_relative.npz",
                        )
                    ),
                )
                self.decoder.parameter_head.apply(self.init_weights)
            else:
                self.decoder = ModeQueryEndpointPCADecoder(
                    token_dim=self.token_dim + self.scene_dim,
                    query_dim=int(
                        getattr(decoder_cfg, "QUERY_DIM", 24)
                    ),
                    pred_len=self.pred_len,
                    num_modes=int(
                        getattr(decoder_cfg, "NUM_MODES", 1)
                    ),
                    num_components=int(
                        getattr(decoder_cfg, "PCA_COMPONENTS", 6)
                    ),
                    basis_path=str(
                        getattr(
                            decoder_cfg,
                            "ENDPOINT_PCA_BASIS_PATH",
                            "assets/pca/"
                            "endpoint_residual_relative.npz",
                        )
                    ),
                    num_heads=int(
                        getattr(decoder_cfg, "QUERY_HEADS", 4)
                    ),
                    ffn_dim=int(
                        getattr(decoder_cfg, "QUERY_FFN_DIM", 80)
                    ),
                    dropout=float(
                        getattr(decoder_cfg, "QUERY_DROPOUT", 0.1)
                    ),
                    use_mode_self_attention=bool(
                        getattr(
                            decoder_cfg,
                            "USE_MODE_SELF_ATTENTION",
                            False,
                        )
                    ),
                    num_layers=int(
                        getattr(decoder_cfg, "QUERY_LAYERS", 1)
                    ),
                    use_residual_refiner=bool(
                        getattr(
                            decoder_cfg,
                            "USE_RESIDUAL_REFINER",
                            False,
                        )
                    ),
                    refiner_hidden_dim=int(
                        getattr(
                            decoder_cfg, "REFINER_HIDDEN_DIM", 64
                        )
                    ),
                    refiner_max_offset=float(
                        getattr(
                            decoder_cfg, "REFINER_MAX_OFFSET", 0.35
                        )
                    ),
                    score_hidden_dim=int(
                        getattr(
                            decoder_cfg, "SCORE_HIDDEN_DIM", 64
                        )
                    ),
                )
            torch.set_rng_state(post_linear_decoder_rng_state)
        else:
            raise ValueError(
                f"Unsupported decoder variant: {self.decoder_variant}"
            )
        self.last_mode_logits = None

        # D1 is deliberately appended after the original ASTRA/S2 modules so
        # their initialization order stays unchanged.  Restoring the CPU RNG
        # also prevents this optional branch from changing later shuffle and
        # dropout randomness in the controlled comparison.
        if self.build_self_state:
            cpu_rng_state = torch.get_rng_state()
            state_hidden_dim = int(
                getattr(structure_cfg, "SELF_STATE_HIDDEN_DIM", 32)
            )
            self.self_state_speed_scale = float(
                getattr(structure_cfg, "SELF_STATE_SPEED_SCALE", 1.5)
            )
            self.self_state_acc_scale = float(
                getattr(structure_cfg, "SELF_STATE_ACC_SCALE", 2.0)
            )
            self.self_state_encoder = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, state_hidden_dim),
                nn.GELU(),
                nn.Linear(state_hidden_dim, self.token_dim),
            )
            self.self_state_encoder.apply(self.init_weights)
            self.self_state_gate = nn.Parameter(torch.zeros(()))
            torch.set_rng_state(cpu_rng_state)
            if not self.use_self_state:
                self.self_state_gate.requires_grad = False
                for parameter in self.self_state_encoder.parameters():
                    parameter.requires_grad = False
        else:
            self.self_state_encoder = None

        # A0/A1 build exactly the same edge-message architecture. The A0
        # control zeros only its four structural inputs, so parameter count,
        # initialization, graph, and optimization interface stay matched.
        if self.build_edge_drwse:
            edge_hidden_dim = int(
                getattr(structure_cfg, "EDGE_DRWSE_HIDDEN_DIM", 8)
            )
            self.edge_drwse = EdgeAlternativePathDRWSE(
                token_dim=self.token_dim,
                hidden_dim=edge_hidden_dim,
                zero_control=self.edge_drwse_zero_control,
            )
        else:
            self.edge_drwse = None
        self.last_edge_drwse_stats = None
   
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)

    def __len__(self):
        pass
    
    def scene_encoder(self, unet_features, temporal_encodings):
        unet_proj = self.lin_proj_unet(unet_features)
        # ic(unet_proj)
        ic(unet_proj.shape)                  # unet_output: (Batch, Frames, Agents, unet_dim)
        unet_trans_input = torch.cat([unet_proj, temporal_encodings], dim=-1)
        unet_trans_input = unet_trans_input.view(
            unet_features.shape[0], -1, self.scene_dim
        )
        # ic(unet_trans_input)
        ic(unet_trans_input.shape)           # unet_trans_input: (Batch, Frames, Agents, unet_dim)
        scene_encodings = self.scene_aware_trans_encoder(unet_trans_input, mask = self.mask)
        # ic(scene_encodings)
        ic(scene_encodings.shape)             # scene_encodings: (Batch, Frames, Agents, unet_dim+temp_dim)
        return scene_encodings
        
    def xy_encoder(self, xy_coords):
        spatial_output = self.lin_proj_xy(xy_coords)
        return spatial_output
    
    def velo_encoder(self, past_loc):
        differences = past_loc[:, :, 1:, :] - past_loc[:, :, :-1, :] 
        velocities = torch.sqrt((differences ** 2).sum(-1) + 1e-6)
        first_frame_velo = torch.zeros(past_loc.shape[0], past_loc.shape[1], 1, device=past_loc.device)
        velocities = torch.cat([first_frame_velo, velocities], dim=2)
        velocities = velocities.permute(0, 2, 1).unsqueeze(-1)
        # ic(velocities)
        ic(velocities.shape)                  # velocities: (Batch, Frames, Agents, 1)
        velo_output = self.lin_proj_velo(velocities)
        return velo_output

    def self_state_features(self, past_loc):
        """Return time-aligned [vx, vy, ax, ay, cos(h), sin(h)]."""
        frequency = float(self.cfg.DATA.FREQUENCY)
        velocity = torch.zeros_like(past_loc)
        velocity[:, :, 1:] = (
            past_loc[:, :, 1:] - past_loc[:, :, :-1]
        ) * frequency
        if past_loc.shape[2] > 1:
            velocity[:, :, 0] = velocity[:, :, 1]

        acceleration = torch.zeros_like(past_loc)
        acceleration[:, :, 1:] = (
            velocity[:, :, 1:] - velocity[:, :, :-1]
        ) * frequency
        if past_loc.shape[2] > 1:
            acceleration[:, :, 0] = acceleration[:, :, 1]

        speed = torch.linalg.vector_norm(
            velocity, dim=-1, keepdim=True
        )
        heading = velocity / speed.clamp(min=1e-6)
        stationary = speed <= 1e-6
        heading = torch.where(stationary, torch.zeros_like(heading), heading)
        features = torch.cat(
            [
                velocity / self.self_state_speed_scale,
                acceleration / self.self_state_acc_scale,
                heading,
            ],
            dim=-1,
        )
        return features.permute(0, 2, 1, 3)

    def self_state_tokens(self, past_loc):
        features = self.self_state_features(past_loc)
        encoded = torch.tanh(self.self_state_encoder(features))
        return encoded.reshape(
            past_loc.shape[0],
            self.obs_len * self.num_pedestrians,
            self.token_dim,
        )
    
    def create_distance_adjacency_matrix(self, past_loc):
        adjacency_matrix = torch.cdist(past_loc, past_loc, p=2.0)
        adjacency_matrix = torch.reciprocal(adjacency_matrix) 
        identity_matrix = torch.eye(self.num_pedestrians).bool().unsqueeze(0).unsqueeze(1).to(adjacency_matrix.device)
        mask = ~identity_matrix
        masked_tensor = adjacency_matrix.masked_select(mask.unsqueeze(0))
        adjacency_matrix = masked_tensor.view(
            past_loc.shape[0],
            self.obs_len,
            self.num_pedestrians * (self.num_pedestrians - 1),
        )
        adjacency_matrix[adjacency_matrix == float('inf')] = 0 # replace inf with 0
        return adjacency_matrix

    def social_encoder(self, past_loc, adjacency=None):
        nodes = past_loc
        if adjacency is None or self.graph_builder.mode == "legacy":
            skeleton = torch.zeros(
                nodes.shape[0],
                nodes.shape[2],
                nodes.shape[2],
                dtype=torch.bool,
                device=nodes.device,
            )
            if self.edge_index.numel():
                edge_index = self.edge_index.to(nodes.device)
                skeleton[:, edge_index[0], edge_index[1]] = True
        else:
            skeleton = self.graph_builder.rwpe_skeleton(adjacency)
        eye = torch.eye(
            nodes.shape[2], dtype=torch.bool, device=nodes.device
        ).unsqueeze(0)
        skeleton = skeleton & ~eye
        distances = torch.cdist(nodes, nodes)
        weights = torch.where(
            skeleton.unsqueeze(1),
            torch.reciprocal(distances + 1e-6),
            torch.zeros_like(distances),
        )
        transition = weights / weights.sum(dim=-1, keepdim=True).clamp(
            min=1e-6
        )
        random_walk = transition
        encodings = []
        for _ in range(self.walk_length):
            encodings.append(torch.diagonal(random_walk, dim1=-2, dim2=-1))
            random_walk = torch.matmul(random_walk, transition)
        social_encodings = torch.stack(encodings, dim=-1)
        return self.lin_proj_social(social_encodings)

    def temporal_encoder(self, batch_size):
        position = torch.arange(0, self.obs_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.temp_dim, 2).float() * (-math.log(10000.0) / self.temp_dim))

        # Calculate sine and cosine components of positional encoding
        pos_enc = torch.zeros(self.obs_len, self.temp_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)
        pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, self.num_pedestrians, -1).to(self.device)
        return pos_enc
    
    def future_temporal_encoder(self, batch_size):
        position = torch.arange(self.obs_len, self.obs_len + self.pred_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.temp_dim, 2).float() * (-math.log(10000.0) / self.temp_dim))

        # Calculate sine and cosine components of positional encoding
        pos_enc = torch.zeros(self.pred_len, self.temp_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)
        pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, self.num_pedestrians, -1).to(self.device)
        return pos_enc

    def aggregate(self, spatial_output, velo_output, social_output, temporal_output):        
        outputs = []
        outputs.append(spatial_output)
        outputs.append(temporal_output)
        if self.cfg.MODEL.INC_VELO:
            outputs.append(velo_output)
        if self.use_rwpe:
            outputs.append(social_output)
        output = torch.cat(outputs, dim=-1)                    
        output = output.view(spatial_output.shape[0], -1, self.token_dim)
        return output

    def group_tokens_by_agent(self, tokens):
        """Convert time-major [B, T*N, D] tokens to [B, N, T*D]."""
        batch_size, token_count, feature_dim = tokens.shape
        if token_count % self.num_pedestrians:
            raise ValueError(
                f"{token_count} tokens cannot be divided among "
                f"{self.num_pedestrians} agents"
            )
        sequence_length = token_count // self.num_pedestrians
        return (
            tokens.reshape(
                batch_size,
                sequence_length,
                self.num_pedestrians,
                feature_dim,
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, self.num_pedestrians, -1)
        )

    def create_custom_mask(self):
        mask = torch.full((self.obs_len * self.num_pedestrians, self.obs_len * self.num_pedestrians), float('-inf'))
        for frame in range(self.obs_len):
            for pedestrian in range(self.num_pedestrians):
                index = frame * self.num_pedestrians + pedestrian
                mask[index, pedestrian::self.num_pedestrians] = 0.0
        return mask.to(self.device)
    
    def forward(
        self,
        past_loc,
        fut_loc,
        unet_features,
        mode='train',
        num_valid=None,
        graph_loc=None,
    ):
        runtime_batch_size = past_loc.shape[0]

        # Spatial Encodings (x, y)
        spatial_features = past_loc.permute(0, 2, 1, 3) 
        # ic(spatial_features)
        ic(spatial_features.shape)           # spatial_features: (Batch, Frames, Agents, 2)
        spatial_encodings = self.xy_encoder(spatial_features)
        # ic(spatial_encodings)
        ic(spatial_encodings.shape)          # spatial_encodings: (Batch, Frames, Agents, spatial_dim)
        
        # Velocity Encodings
        if self.cfg.MODEL.INC_VELO:
            velo_encodings = self.velo_encoder(past_loc)
            # ic(velo_encodings)
            ic(velo_encodings.shape)         # velo_encodings: (Batch, Frames, Agents, velo_dim)
        else:
            velo_encodings = None 
        
        # Temporal Encodings
        temporal_encodings = self.temporal_encoder(runtime_batch_size)

        # Stage-1 graph: a single history-conditioned graph is shared across
        # all observed tokens. Legacy mode reproduces ASTRA's temporal-only
        # Transformer mask and original fixed RWPE graph.
        if graph_loc is None:
            graph_loc = past_loc
        graph_features = graph_loc.permute(0, 2, 1, 3)
        graph_context = None
        drse_output = None
        if self.drse_enabled:
            graph_context = self.graph_builder.build_context(
                graph_features, num_valid
            )
            graph_adjacency = graph_context.adjacency
            valid_agents = graph_context.valid
            drse_output = self.drse_encoder(graph_context)
            self.last_drse_stats = {
                "node_embedding_abs": (
                    drse_output.node_embedding.detach().abs().mean()
                ),
                "kinematic_bias_abs": (
                    drse_output.kinematic_bias.detach().abs().mean()
                ),
                "structure_bias_abs": (
                    drse_output.structure_bias.detach().abs().mean()
                ),
            }
        else:
            graph_adjacency, valid_agents = self.graph_builder.build(
                graph_features, num_valid
            )
            self.last_drse_stats = None
        self.last_graph_stats = self.graph_builder.stats(
            graph_adjacency, valid_agents
        )
        if self.graph_builder.mode == "legacy":
            agent_mask = self.mask
        else:
            agent_mask = self.graph_builder.attention_mask(
                graph_adjacency,
                self.obs_len,
                self.cfg.MODEL.NHEAD,
                edge_bias=(
                    drse_output.edge_bias
                    if drse_output is not None
                    else None
                ),
            )

        if mode == 'train' and self.cfg.MODEL.USE_VAE:
            spatial_features_y = fut_loc.permute(0, 2, 1, 3)
            spatial_encodings_y = self.xy_encoder(spatial_features_y)
            temporal_encodings_y = self.future_temporal_encoder(
                runtime_batch_size
            )
            ic(spatial_encodings_y.shape)
            ic(temporal_encodings_y.shape)
            encodings_y = torch.cat([spatial_encodings_y, temporal_encodings_y], dim=-1)
            encodings_x = torch.cat([spatial_encodings, temporal_encodings], dim=-1)
            encodings_y = self.group_tokens_by_agent(encodings_y)
            encodings_x = self.group_tokens_by_agent(encodings_x)
            ic(encodings_x.shape, encodings_y.shape)
            mean_c, log_var_c, MLP_input_c = self.cvae_model(encodings_x, encodings_y)
            ic(mean_c.shape)
            ic(log_var_c.shape)
            ic(MLP_input_c.shape)
        else:
            mean_c, log_var_c, MLP_input_c = None, None, None

        # ic(temporal_encodings)
        ic(temporal_encodings.shape)        # temporal_encodings: (Batch, Frames, Agents, temp_dim)
        
        # Scene Encodings
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            unet_features = unet_features.permute(0, 2, 1, 3)
            # ic(unet_features)
            ic(unet_features.shape)          # unet_features: (Batch, Frames, Agents, feature_dim)
            scene_encodings = self.scene_encoder(unet_features, temporal_encodings)
        else:
            scene_encodings = None
            
        # Social Encodings
        if self.use_rwpe:
            social_encodings = self.social_encoder(
                graph_features, graph_adjacency
            )
            # ic(social_encodings)
            ic(social_encodings.shape)             # social_encodings: (Batch, Frames, Agents, walk_length)        
        else:
            social_encodings = None
                
        # Agent Aware Transformer Encoder 
        trans_input = self.aggregate(spatial_encodings, velo_encodings, social_encodings, temporal_encodings)   
        if self.use_self_state:
            state_tokens = self.self_state_tokens(past_loc)
            state_gate = torch.tanh(self.self_state_gate)
            trans_input = trans_input + state_gate * state_tokens
            if self.last_drse_stats is not None:
                self.last_drse_stats["self_state_embedding_abs"] = (
                    (state_gate * state_tokens).detach().abs().mean()
                )
        if drse_output is not None:
            node_tokens = (
                drse_output.node_embedding[:, None, :, :]
                .expand(
                    -1,
                    self.obs_len,
                    -1,
                    -1,
                )
                .reshape(
                    runtime_batch_size,
                    self.obs_len * self.num_pedestrians,
                    self.token_dim,
                )
            )
            trans_input = trans_input + node_tokens
        if self.use_edge_drwse:
            edge_drwse_output = self.edge_drwse(
                trans_input, graph_adjacency
            )
            trans_input = trans_input + edge_drwse_output.residual
            self.last_edge_drwse_stats = edge_drwse_output.stats
            if self.last_drse_stats is not None:
                self.last_drse_stats.update(
                    {
                        "edge_drwse_gate": (
                            edge_drwse_output.stats["gate"]
                        ),
                        "edge_drwse_feature_abs": (
                            edge_drwse_output.stats["feature_abs_mean"]
                        ),
                        "edge_drwse_alternative_edge_rate": (
                            edge_drwse_output.stats[
                                "alternative_edge_rate"
                            ]
                        ),
                        "edge_drwse_residual_abs": (
                            edge_drwse_output.stats["residual_abs_mean"]
                        ),
                    }
                )
        else:
            self.last_edge_drwse_stats = None
        # ic(trans_input)     
        ic(trans_input.shape)                      # trans_input: (Batch, Frames*Agents, token_dim)
        if self.cfg.MODEL.TRANS_MASK:
            ic(self.mask)
            ic(self.mask.shape)                    # mask: (Frames*Agents, Frames*Agents)
        trans_encodings = self.agent_aware_trans_encoder(
            trans_input, mask=agent_mask
        )
        # ic(trans_encodings)
        ic(trans_encodings.shape)                  # trans_encodings: (Batch, Frames*Agents, token_dim)                             
        
        # MLP Inputs
        MLP_inputs = [] 
        MLP_inputs.append(trans_encodings)
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            MLP_inputs.append(scene_encodings)
        MLP_input = torch.cat(MLP_inputs, dim=-1)
        # ic(MLP_input)
        ic(MLP_input.shape)                               # MLP_input: (Batch, Frames*Agents, token_dim + unet_dim

        # if self.cfg.MODEL.USE_VAE:
        #     ic(spatial_features.shape, spatial_features)
        #     past_loc_c = spatial_features.view(self.batch_size_device,  -1, 2)
        #     ic(MLP_input.shape, MLP_input)
        #     MLP_input = torch.cat([MLP_input, past_loc_c], dim = -1)
        #     ic(MLP_input.shape, MLP_input)
        
        if self.decoder_variant == "query_endpoint_pca":
            encoded_tokens = (
                MLP_input.reshape(
                    runtime_batch_size,
                    self.obs_len,
                    self.num_pedestrians,
                    self.token_dim + self.scene_dim,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            model_output, self.last_mode_logits = self.decoder(
                encoded_tokens, past_loc
            )
            return (
                None,
                None,
                model_output,
                None,
                None,
                None,
            )

        self.last_mode_logits = None
        MLP_input = self.group_tokens_by_agent(MLP_input)
        # ic(MLP_input)
        ic(MLP_input.shape)                               # MLP_input: (Batch, Agents, Frames*(token_dim + unet_dim + temp_dim))
        if self.cfg.MODEL.USE_VAE:
            mean, log_var, MLP_input = self.vae_model(MLP_input)
            
            ic(MLP_input.shape)
        else:
            mean, log_var = None, None

        # Non-autoregressive trajectory decoding.
        if self.decoder_variant in {"pca", "endpoint_pca"}:
            model_output = self.decoder(MLP_input, past_loc)
        else:
            MLP_decodings = self.decoder(MLP_input)
            if self.cfg.MODEL.USE_VAE:
                model_output = MLP_decodings.view(
                    runtime_batch_size,
                    self.num_pedestrians,
                    self.num_samples,
                    self.pred_len,
                    self.output_dim,
                )
            else:
                model_output = MLP_decodings.view(
                    runtime_batch_size,
                    self.num_pedestrians,
                    self.pred_len,
                    self.output_dim,
                )
        
        # ic(model_output)
        ic(model_output.shape)                            # model_output: (Batch, Agents, K, Pred_len, output_dim)
        if mode == 'train' and self.cfg.MODEL.USE_VAE:
            c_mlp_decodings = self.decoder(MLP_input_c)
            c_model_output = c_mlp_decodings.view(
                runtime_batch_size,
                self.num_pedestrians,
                1,
                self.pred_len,
                self.output_dim,
            )
            return mean, log_var, model_output, mean_c, log_var_c, c_model_output
        return mean, log_var, model_output, None, None, None
