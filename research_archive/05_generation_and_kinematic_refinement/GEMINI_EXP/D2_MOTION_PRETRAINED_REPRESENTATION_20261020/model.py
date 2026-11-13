import random
import numpy as np
import torch
import torch.nn as nn

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=8):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, 8, d_model)
        return x + self.pe[None, :, :]

class MotionPretrainedEncoder(nn.Module):
    """
    2-layer TransformerEncoder for self-supervised motion pretraining.
    Input: (B, 8, 4) -> [rel_x, rel_y, mask, t/8]
    Frozen in downstream stage to provide h64 representation.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 64)
        self.pos = PositionalEncoding(64, 8)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            activation='gelu',
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.readout = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward_pretrain(self, x):
        h = self.pos(self.proj(x))
        out = self.transformer(h)
        pred = self.readout(out) # (B, 8, 2)
        return out, pred

    @torch.no_grad()
    def extract_representation(self, past_tensor):
        # past_tensor: (B, 8, 2)
        # Local origin is past[:, 0:1, :]
        B = len(past_tensor)
        rel_pos = past_tensor - past_tensor[:, 0:1, :]
        mask = torch.ones((B, 8, 1), dtype=torch.float32, device=past_tensor.device)
        t_norm = (torch.arange(8, dtype=torch.float32, device=past_tensor.device) / 8.0)[None, :, None].expand(B, -1, -1)
        x = torch.cat((rel_pos.float(), mask, t_norm), dim=-1) # (B, 8, 4)
        h = self.pos(self.proj(x))
        out = self.transformer(h) # (B, 8, 64)
        # Return last token (t=7)
        h64 = out[:, 7, :].detach()
        return h64

class PretrainedConditionedGenerator(nn.Module):
    """
    Downstream K20 Candidate Generator conditioned on [P2 (96), h64 (64)] = 160.
    Total trainable parameters: 40,328.
    """
    def __init__(self, stats):
        super().__init__()
        for key in ('p2_mean', 'p2_std', 'v_mean', 'v_std', 'z_mean', 'z_std', 'centers12'):
            self.register_buffer(key, torch.tensor(stats[key], dtype=torch.float32))
        self.register_buffer('s', torch.arange(1, 13, dtype=torch.float64) / 12.0)

        # 1. Context: 160 -> 128 -> 64 GELU
        self.context = nn.Sequential(
            nn.Linear(160, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU()
        )

        # 2. Query
        self.query = nn.Parameter(torch.empty(20, 16))
        nn.init.normal_(self.query, 0.0, 0.02)

        # 3. Goal head: 80 -> 64 -> 2 GELU
        self.goal_head = nn.Sequential(
            nn.Linear(80, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        nn.init.zeros_(self.goal_head[-1].weight)
        nn.init.zeros_(self.goal_head[-1].bias)

        # 4. Process head: 84 -> 64 -> 6 GELU
        self.process_head = nn.Sequential(
            nn.Linear(84, 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )
        nn.init.zeros_(self.process_head[-1].weight)
        nn.init.zeros_(self.process_head[-1].bias)

    def encode(self, p2, h64):
        p2_norm = (p2 - self.p2_mean) / self.p2_std
        feat_160 = torch.cat((p2_norm, h64), dim=-1) # (B, 160)
        h = self.context(feat_160) # (B, 64)
        hq = torch.cat((h[:, None, :].expand(-1, 20, -1), self.query[None, :, :].expand(len(p2), -1, -1)), dim=-1) # (B, 20, 80)
        return hq

    def decode_with_goal(self, hq, v0, z):
        B = len(hq)
        z_norm = ((z - self.z_mean) / self.z_std).to(hq.dtype)
        v0_norm = ((v0 - self.v_mean) / self.v_std).to(hq.dtype)[:, None, :].expand(-1, 20, -1)
        x_proc = torch.cat((hq, z_norm, v0_norm), dim=-1) # (B, 20, 84)

        abc = self.process_head(x_proc).reshape(B, 20, 3, 2).double()
        a = abc[:, :, 0, :][:, :, None, :]
        b = abc[:, :, 1, :][:, :, None, :]
        c = abc[:, :, 2, :][:, :, None, :]

        s = self.s[None, None, :, None]
        v = v0.double()[:, None, None, :]
        zz = z.double()[:, :, None, :]

        path = 4.8 * s * v + s.square() * (zz - 4.8 * v + a) + s.square() * (1 - s) * b + s.square() * (1 - s) * (2 * s - 1) * c
        return dict(
            path=path,
            goal=z,
            process_coeff=abc
        )

    def forward(self, p2, h64, v0):
        hq = self.encode(p2, h64)
        delta = self.goal_head(hq) # (B, 20, 2)
        base = 4.8 * v0[:, None, :] + self.centers12[None, :, :]
        z = base + delta.double()
        return self.decode_with_goal(hq, v0, z)

def compute_pretrain_loss(pred, past, arm):
    """
    Pred: (B, 8, 2)
    Past: (B, 8, 2)
    Local origin: past[:, 0:1, :]
    Arm A: hidden [4, 5]
    Arm B: hidden [6, 7]
    """
    rel = past - past[:, 0:1, :]
    if arm == 'A':
        target = rel[:, [4, 5], :]
        pred_sub = pred[:, [4, 5], :]
    elif arm == 'B':
        target = rel[:, [6, 7], :]
        pred_sub = pred[:, [6, 7], :]
    else:
        raise ValueError(f"Unknown arm: {arm}")

    err = torch.linalg.vector_norm(pred_sub - target, dim=-1) # (B, 2)
    loss = err.mean()
    return loss

def compute_generator_loss(out, y12):
    p = out['path']
    z = out['goal']
    B = len(p)

    d = torch.linalg.vector_norm(p - y12[:, None], dim=-1)
    ade = d.mean(-1)
    fde = d[:, :, -1]
    q = ade + 0.5 * fde
    winner = q.argmin(1)
    row = torch.arange(B, device=p.device)
    loss_path = q[row, winner].mean()

    # Goal loss to terminal GT
    target_goal = y12[:, -1, :] # (B, 2)
    winner_goal = z[row, winner]
    loss_goal = torch.linalg.vector_norm(winner_goal - target_goal, dim=-1).mean()

    # Diversity repulsion
    a_idx, b_idx = torch.triu_indices(20, 20, 1, device=p.device)
    pd = torch.linalg.vector_norm(p[:, a_idx] - p[:, b_idx], dim=-1).mean(-1)
    loss_div = torch.relu(0.30 - pd).square().mean()

    total = loss_path + 0.5 * loss_goal + 0.05 * loss_div
    return total, dict(
        total=float(total.item()),
        loss_path=float(loss_path.item()),
        loss_goal=float(loss_goal.item()),
        loss_div=float(loss_div.item()),
        winner=winner
    )
