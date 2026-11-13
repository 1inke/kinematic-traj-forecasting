import os, random
import numpy as np
import torch
import torch.nn as nn

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_kernels():
    """
    bank[0]: free position kernel
    bank[1]: integrated acceleration kernel
    Displacement bound: ||r[t]|| <= 0.20 * (t / 12)^2
    """
    bank = np.zeros((2, 12, 12), np.float64)
    for t in range(1, 13):
        bank[0, t - 1, t - 1] = 0.20 / 144 * t * t
        for j in range(1, t + 1):
            bank[1, t - 1, j - 1] = 0.20 / 144 * (2 * (t - j) + 1)
    return bank

class StandardRefiner(nn.Module):
    """
    Standard candidate refiner network:
    Input: [p2 (96), candidate_feat (24)] = 120
    Architecture: Linear(120, 128) -> GELU -> Linear(128, 64) -> GELU -> Linear(64, 24)
    Final layer zero-initialized (starts at pure native identity).
    Output: control u in unit disk -> integrated acceleration correction r -> p = native + r.
    """
    def __init__(self, st, representation='position'):
        super().__init__()
        assert representation in ('position', 'velocity')
        self.representation = representation

        self.register_buffer('mean', torch.from_numpy(st['mean'].copy()))
        self.register_buffer('std', torch.from_numpy(st['std'].copy()))
        self.register_buffer('kernelbank', torch.from_numpy(compute_kernels()))

        self.head = nn.Sequential(
            nn.Linear(120, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 24)
        )
        # Zero-initialize final layer so refiner starts as identity transformation
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def extract_features(self, p2, native):
        # native: (B, 20, 12, 2)
        if self.representation == 'position':
            cand_feat = native.reshape(-1, 20, 24).to(p2.dtype)
        elif self.representation == 'velocity':
            v = torch.diff(native, dim=2, prepend=torch.zeros_like(native[:, :, :1])) / 0.4
            cand_feat = v.reshape(-1, 20, 24).to(p2.dtype)
        # Expand p2 across 20 slots
        p2_expanded = p2[:, None].expand(-1, 20, -1)
        raw = torch.cat((p2_expanded, cand_feat), dim=-1) # (B, 20, 120)
        return raw

    def forward(self, p2, native):
        raw = self.extract_features(p2, native)
        # Normalize input features
        norm_in = ((raw.double() - self.mean) / self.std).to(p2.dtype)
        z = self.head(norm_in).reshape(-1, 20, 12, 2) # (B, 20, 12, 2)
        
        # Bounded control u in unit disk
        u = torch.tanh(z).double()
        u_norm = torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1.0)
        u = u / u_norm

        # Double integration for acceleration
        # kernelbank[1] has shape (12, 12)
        r = torch.einsum('tj, bkjd -> bktd', self.kernelbank[1], u)
        p = native.double() + r
        return p, z, u, r

def generate_predictions(net, data, indices, batch_size=128):
    """Generates full trajectory predictions on CPU arrays."""
    net.eval()
    p_list, raw_list, u_list, r_list = [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(indices), batch_size):
            b_idx = indices[i:i + batch_size]
            p2_b = torch.from_numpy(data['p2'][b_idx]).to(DEVICE)
            nat_b = torch.from_numpy(data['native'][b_idx]).to(DEVICE)
            p_out, z_out, u_out, r_out = net(p2_b, nat_b)
            p_list.append(p_out.cpu().numpy())
            raw_list.append(z_out.cpu().numpy())
            u_list.append(u_out.cpu().numpy())
            r_list.append(r_out.cpu().numpy())

    p = np.concatenate(p_list, axis=0)
    raw = np.concatenate(raw_list, axis=0)
    u = np.concatenate(u_list, axis=0)
    r = np.concatenate(r_list, axis=0)
    return p, raw, u, r
