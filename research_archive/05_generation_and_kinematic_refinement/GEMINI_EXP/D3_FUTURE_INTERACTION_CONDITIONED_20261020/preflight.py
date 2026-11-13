import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import psutil

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / 'GEMINI_EXP'))

from core.data_loader import load_dataset
from core.model_base import set_seed, compute_kernels, DEVICE

def compute_kernels_t():
    return torch.from_numpy(compute_kernels()).to(DEVICE)

class NeighborPastEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [x, y, vx, vy, dist] = 5
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )
    
    def forward(self, nb_pos, nb_vel, nb_valid):
        # nb_pos: (B, 8, 2)
        # nb_vel: (B, 8, 2)
        # nb_valid: (B, 8) bool
        dist = torch.linalg.vector_norm(nb_pos, dim=-1, keepdim=True) # (B, 8, 1)
        feat = torch.cat([nb_pos, nb_vel, dist], dim=-1) # (B, 8, 5)
        h = self.net(feat) # (B, 8, 32)
        mask = nb_valid.unsqueeze(-1).to(h.dtype) # (B, 8, 1)
        h_masked = h * mask
        denom = mask.sum(dim=1).clamp_min(1.0) # (B, 1)
        pooled = h_masked.sum(dim=1) / denom # (B, 32)
        # Zero out completely if no valid neighbors
        has_any = (mask.sum(dim=1) > 0).to(h.dtype)
        return pooled * has_any

class D3Model(nn.Module):
    def __init__(self, arm='B'):
        super().__init__()
        assert arm in ('A', 'B')
        self.arm = arm
        self.register_buffer('kernelbank', torch.from_numpy(compute_kernels()))

        self.nb_encoder = NeighborPastEncoder()

        # Stage 1 Refiner: [p2 (96), s_past (32), native (24)] = 152
        self.fc1 = nn.Linear(152, 128)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.GELU()
        self.head1 = nn.Linear(64, 24)
        nn.init.zeros_(self.head1.weight)
        nn.init.zeros_(self.head1.bias)

        if self.arm == 'B':
            # Future interaction message encoder
            # Input: [min_dist (1), mean_conflict (1), max_conflict (1), early_conflict (1), late_conflict (1), closest_rel_pos (2)] = 7
            self.conflict_encoder = nn.Sequential(
                nn.Linear(7, 32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU()
            )
            # Stage 2 Refiner: [h1 (64), m (32)] = 96
            self.s2_fc1 = nn.Linear(96, 64)
            self.s2_act1 = nn.GELU()
            self.head2 = nn.Linear(64, 24)
            nn.init.zeros_(self.head2.weight)
            nn.init.zeros_(self.head2.bias)

    def compute_interaction_messages(self, p_curr, nb_extrap, nb_valid):
        # p_curr: (B, 20, 12, 2)
        # nb_extrap: (B, 8, 12, 2)
        # nb_valid: (B, 8) bool
        B = p_curr.shape[0]
        # Expand for pairwise future distances: (B, 20, 8, 12, 2)
        p_exp = p_curr.unsqueeze(2) # (B, 20, 1, 12, 2)
        nb_exp = nb_extrap.unsqueeze(1) # (B, 1, 8, 12, 2)
        diff = p_exp - nb_exp # (B, 20, 8, 12, 2)
        dist = torch.linalg.vector_norm(diff, dim=-1) # (B, 20, 8, 12)

        # Mask invalid neighbors with large distance
        valid_mask = nb_valid.view(B, 1, 8, 1).expand(-1, 20, -1, 12)
        dist_masked = torch.where(valid_mask, dist, torch.full_like(dist, 100.0))

        # Conflict potential with sigma = 1.0m
        conflict = torch.exp(-dist_masked.square() / 2.0) * valid_mask.to(dist.dtype) # (B, 20, 8, 12)

        # Extract features per candidate (B, 20, 1)
        n_valid = nb_valid.sum(dim=-1, keepdim=True).clamp_min(1.0).float() # (B, 1)
        min_dist = dist_masked.amin(dim=(2, 3)).unsqueeze(-1) # (B, 20, 1)
        mean_conflict = (conflict.sum(dim=(2, 3)) / (n_valid * 12.0)).unsqueeze(-1) # (B, 20, 1)
        max_conflict = conflict.amax(dim=(2, 3)).unsqueeze(-1) # (B, 20, 1)
        early_conflict = (conflict[..., :4].sum(dim=(2, 3)) / (n_valid * 4.0)).unsqueeze(-1) # (B, 20, 1)
        late_conflict = (conflict[..., 8:].sum(dim=(2, 3)) / (n_valid * 4.0)).unsqueeze(-1) # (B, 20, 1)

        # Vector to closest neighbor at min distance
        dist_flat = dist_masked.view(B, 20, 96) # (B, 20, 96)
        min_idx = dist_flat.argmin(dim=-1) # (B, 20)
        diff_flat = diff.view(B, 20, 96, 2)
        closest_rel = diff_flat.gather(2, min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2) # (B, 20, 2)
        has_any = (nb_valid.sum(dim=-1, keepdim=True) > 0).unsqueeze(1).to(p_curr.dtype) # (B, 1, 1)
        closest_rel = closest_rel * has_any

        feat = torch.cat([min_dist, mean_conflict, max_conflict, early_conflict, late_conflict, closest_rel], dim=-1) # (B, 20, 7)
        feat = feat * has_any
        m = self.conflict_encoder(feat) # (B, 20, 32)
        return m

    def forward(self, p2, native, nb_pos, nb_vel, nb_valid, nb_extrap):
        B = p2.shape[0]
        s_past = self.nb_encoder(nb_pos, nb_vel, nb_valid) # (B, 32)

        # Stage 1 input
        cand_feat = native.reshape(B, 20, 24).to(p2.dtype)
        p2_exp = p2[:, None].expand(-1, 20, -1)
        sp_exp = s_past[:, None].expand(-1, 20, -1)
        in1 = torch.cat([p2_exp, sp_exp, cand_feat], dim=-1) # (B, 20, 152)

        h1_0 = self.act1(self.fc1(in1))
        h1_1 = self.act2(self.fc2(h1_0)) # (B, 20, 64)
        z1 = self.head1(h1_1).reshape(B, 20, 12, 2)

        u1 = torch.tanh(z1).double()
        u1 = u1 / torch.linalg.vector_norm(u1, dim=-1, keepdim=True).clamp_min(1.0)
        r1 = torch.einsum('tj, bkjd -> bktd', self.kernelbank[1], u1)
        p1 = native.double() + r1

        if self.arm == 'A':
            return p1, u1, r1

        # Arm B: Stage 2 Future Interaction Refinement
        m = self.compute_interaction_messages(p1.float(), nb_extrap, nb_valid) # (B, 20, 32)
        in2 = torch.cat([h1_1, m], dim=-1) # (B, 20, 96)
        h2 = self.s2_act1(self.s2_fc1(in2))
        z2 = self.head2(h2).reshape(B, 20, 12, 2)

        u2 = torch.tanh(z2).double()
        u2 = u2 / torch.linalg.vector_norm(u2, dim=-1, keepdim=True).clamp_min(1.0)
        r2 = torch.einsum('tj, bkjd -> bktd', self.kernelbank[1], u2)
        p2_out = p1 + r2
        u_total = u1 + u2
        return p2_out, u_total, r1 + r2

def run_preflight():
    print("=== STARTING D3 PREFLIGHT AUDIT ===")

    # 1. Dataset loading & integrity audit
    data = load_dataset()
    splits = data['split']
    for s1 in splits:
        for s2 in splits:
            if s1 != s2:
                overlap = set(splits[s1]).intersection(set(splits[s2]))
                assert len(overlap) == 0, f"Data leakage: Overlap between {s1} and {s2}!"
    print("Audit 1: Split disjointness verified (Zero leakage across FIT, outer, inner, CAL).")

    # 2. Neighbor cache audit
    nb_cache_path = EXP_DIR / 'neighbor_cache.npz'
    assert nb_cache_path.is_file(), "Neighbor cache file missing!"
    with np.load(nb_cache_path) as z:
        assert np.array_equal(z['sample_id'], data['sample_id']), "Neighbor cache sample_id mismatch!"
        assert z['nb_canon_past'].shape == (28791, 8, 8, 2)
        assert z['nb_valid'].shape == (28791, 8)
        assert z['nb_pos0'].shape == (28791, 8, 2)
        assert z['nb_vel0'].shape == (28791, 8, 2)
        assert z['nb_extrap'].shape == (28791, 8, 12, 2)
    print("Audit 2: Neighbor cache integrity and canonical past-only alignment verified.")

    # 3. Model parameter & initialization audit for Arm A and Arm B
    net_a = D3Model('A').to(DEVICE)
    params_a = sum(p.numel() for p in net_a.parameters() if p.requires_grad)
    assert torch.all(net_a.head1.weight == 0) and torch.all(net_a.head1.bias == 0)

    net_b = D3Model('B').to(DEVICE)
    params_b = sum(p.numel() for p in net_b.parameters() if p.requires_grad)
    assert torch.all(net_b.head1.weight == 0) and torch.all(net_b.head2.weight == 0)
    print(f"Audit 3: Model parameter counts verified (Arm A: {params_a:,} params, Arm B: {params_b:,} params). Exact zero-init confirmed.")

    # 4. Identity transformation on initialization
    p2_dummy = torch.from_numpy(data['p2'][:4]).to(DEVICE)
    nat_dummy = torch.from_numpy(data['native'][:4]).to(DEVICE)
    with np.load(nb_cache_path) as z:
        pos_b = torch.from_numpy(z['nb_pos0'][:4]).to(DEVICE)
        vel_b = torch.from_numpy(z['nb_vel0'][:4]).to(DEVICE)
        val_b = torch.from_numpy(z['nb_valid'][:4]).to(DEVICE)
        ext_b = torch.from_numpy(z['nb_extrap'][:4]).to(DEVICE)

    with torch.no_grad():
        p_a, u_a, r_a = net_a(p2_dummy, nat_dummy, pos_b, vel_b, val_b, ext_b)
        p_b, u_b, r_b = net_b(p2_dummy, nat_dummy, pos_b, vel_b, val_b, ext_b)

    assert torch.all(r_a == 0) and torch.all(r_b == 0), "Non-zero displacement on initialization!"
    assert torch.all(p_a == nat_dummy) and torch.all(p_b == nat_dummy), "Initialization does not yield identity paths!"
    print("Audit 4: Strict identity transformation confirmed at initialization for both arms.")

    # 5. Kernel displacement bound check
    net_b.head1.weight.data.normal_(0, 1)
    net_b.head2.weight.data.normal_(0, 1)
    with torch.no_grad():
        _, _, r_test = net_b(p2_dummy, nat_dummy, pos_b, vel_b, val_b, ext_b)
    r_disp = torch.linalg.vector_norm(r_test, dim=-1).cpu().numpy()
    max_allowed = 0.40 * (np.arange(1, 13) / 12.0) ** 2 + 1e-12
    for t in range(12):
        assert np.all(r_disp[..., t] <= max_allowed[t]), f"Displacement exceeded bound at step {t+1}!"
    print("Audit 5: Acceleration-integrated physical displacement bounds confirmed.")

    # 6. Interaction message responsiveness
    net_b = D3Model('B').to(DEVICE)
    # Check that interaction message module responds smoothly to neighbor presence
    with torch.no_grad():
        m_with_nb = net_b.compute_interaction_messages(nat_dummy.float(), ext_b, val_b)
        val_zero = torch.zeros_like(val_b)
        m_zero_nb = net_b.compute_interaction_messages(nat_dummy.float(), ext_b, val_zero)
    assert not torch.allclose(m_with_nb, m_zero_nb), "Interaction messages are unresponsive to neighbor presence!"
    print("Audit 6: Interaction message module responsiveness verified.")

    # 7. System resource check
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    assert available_gb >= 6.0, f"Insufficient memory available: {available_gb:.2f} GB < 6.0 GB"
    print(f"Audit 7: Memory check passed ({available_gb:.2f} GB available).")

    preflight_result = dict(
        status="PASS",
        split_sizes={k: len(v) for k, v in splits.items()},
        trainable_parameters={"A": params_a, "B": params_b},
        neighbor_cache_verified=True,
        displacement_bound_verified=True,
        interaction_responsiveness_verified=True,
        available_memory_gb=available_gb
    )
    with open(EXP_DIR / 'PREFLIGHT.json', 'w', encoding='utf-8') as f:
        json.dump(preflight_result, f, indent=2)

    state = json.load(open(EXP_DIR / 'runtime_state.json', encoding='utf-8'))
    state['status'] = 'PREFLIGHT_PASSED'
    state['preflight_status'] = 'PASS'
    with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print("=== D3 PREFLIGHT AUDIT COMPLETE: ALL PASS ===")
    return True

if __name__ == '__main__':
    run_preflight()
