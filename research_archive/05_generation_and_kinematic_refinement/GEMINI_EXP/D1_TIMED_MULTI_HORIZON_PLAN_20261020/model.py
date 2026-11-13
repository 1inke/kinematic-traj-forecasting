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

class TimedMultiHorizonModel(nn.Module):
    """
    D1: Timed Multi-Horizon Plan Supervision Model
    Context: 96 -> 128 -> 64 GELU
    Query: 20 x 16 (N(0, 0.02))
    Plan Head: 80 -> 64 -> 8 GELU (outputs 4 2D plan variables, zero-init)
    Process Head: 90 -> 64 -> 6 GELU (inputs hq_80 + plan_8 + v0_2, zero-init)
    Total parameters: 32,910
    """
    def __init__(self, stats):
        super().__init__()
        for key in ('p2_mean', 'p2_std', 'v_mean', 'v_std', 'z_mean', 'z_std', 'centers12'):
            self.register_buffer(key, torch.tensor(stats[key], dtype=torch.float32))
        self.register_buffer('s', torch.arange(1, 13, dtype=torch.float64) / 12.0)

        # 1. Context Encoder
        self.context = nn.Sequential(
            nn.Linear(96, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU()
        )

        # 2. Query Embeddings
        self.query = nn.Parameter(torch.empty(20, 16))
        nn.init.normal_(self.query, 0.0, 0.02)

        # 3. Plan Head (4 points x 2D = 8)
        self.plan_head = nn.Sequential(
            nn.Linear(80, 64),
            nn.GELU(),
            nn.Linear(64, 8)
        )
        nn.init.zeros_(self.plan_head[-1].weight)
        nn.init.zeros_(self.plan_head[-1].bias)

        # 4. Process Head (80 + 8 + 2 = 90 -> 64 -> 6)
        self.process_head = nn.Sequential(
            nn.Linear(90, 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )
        nn.init.zeros_(self.process_head[-1].weight)
        nn.init.zeros_(self.process_head[-1].bias)

    def encode(self, p2):
        assert p2.ndim == 2 and p2.shape[1] == 96
        h = self.context((p2 - self.p2_mean) / self.p2_std)
        hq = torch.cat((h[:, None, :].expand(-1, 20, -1), self.query[None, :, :].expand(len(p2), -1, -1)), dim=-1)
        return hq

    def decode_with_plan(self, hq, v0, plan):
        B = len(hq)
        plan_norm = ((plan - self.z_mean) / self.z_std).to(hq.dtype).reshape(B, 20, 8)
        v0_norm = ((v0 - self.v_mean) / self.v_std).to(hq.dtype)[:, None, :].expand(-1, 20, -1)

        x_proc = torch.cat((hq, plan_norm, v0_norm), dim=-1)
        abc = self.process_head(x_proc).reshape(B, 20, 3, 2).double()
        a = abc[:, :, 0, :][:, :, None, :]
        b = abc[:, :, 1, :][:, :, None, :]
        c = abc[:, :, 2, :][:, :, None, :]

        s = self.s[None, None, :, None]
        v = v0.double()[:, None, None, :]
        z4 = plan[:, :, 3, :][:, :, None, :]

        path = 4.8 * s * v + s.square() * (z4 - 4.8 * v + a) + s.square() * (1 - s) * b + s.square() * (1 - s) * (2 * s - 1) * c
        return dict(
            path=path,
            plan=plan,
            process_coeff=abc,
            z4=z4.squeeze(2)
        )

    def forward(self, p2, v0):
        B = len(p2)
        hq = self.encode(p2)
        delta = self.plan_head(hq).reshape(B, 20, 4, 2)
        base = 4.8 * v0[:, None, None, :] + self.centers12[None, :, None, :]
        plan = base + delta.double()
        return self.decode_with_plan(hq, v0, plan)

def compute_objective(out, y12, y20, m20, arm):
    p = out['path']
    plan = out['plan']
    B = len(p)

    # 1. Path quality Q = ADE + 0.5 * FDE
    d = torch.linalg.vector_norm(p - y12[:, None], dim=-1)
    ade = d.mean(-1)
    fde = d[:, :, -1]
    q = ade + 0.5 * fde
    winner = q.argmin(1)
    row = torch.arange(B, device=p.device)
    loss_path = q[row, winner].mean()

    # 2. Plan supervision loss
    if bool(m20.any()):
        m_idx = m20
        plan_sub = plan[row, winner][m_idx]
        if arm == 'A':
            y20_sub = 0.6 * y20[m_idx]
            target_sub = y20_sub[:, None, :].expand(-1, 4, -1)
        elif arm == 'B':
            y4_sub = 3.0 * y12[m_idx, 3, :]
            y8_sub = 1.5 * y12[m_idx, 7, :]
            y12_sub = 1.0 * y12[m_idx, 11, :]
            y20_sub = 0.6 * y20[m_idx]
            target_sub = torch.stack((y4_sub, y8_sub, y12_sub, y20_sub), dim=1)
        else:
            raise ValueError(f"Unknown arm: {arm}")

        plan_err = torch.linalg.vector_norm(plan_sub - target_sub, dim=-1).mean(-1)
        loss_plan = plan_err.sum() / B
    else:
        loss_plan = p.sum() * 0.0

    # 3. Diversity repulsion
    a_idx, b_idx = torch.triu_indices(20, 20, 1, device=p.device)
    pd = torch.linalg.vector_norm(p[:, a_idx] - p[:, b_idx], dim=-1).mean(-1)
    loss_div = torch.relu(0.30 - pd).square().mean()

    total_loss = loss_path + 0.5 * loss_plan + 0.05 * loss_div
    return total_loss, dict(
        total=float(total_loss.item()),
        loss_path=float(loss_path.item()),
        loss_plan=float(loss_plan.item()),
        loss_div=float(loss_div.item()),
        winner=winner,
        ties=(q == q.min(1, keepdim=True).values).sum(1)
    )

def create_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False
    )
